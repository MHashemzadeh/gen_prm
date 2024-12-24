from typing import Type, Callable, Optional, Literal
import numpy as np
from reasoners.benchmark import GSM8KEvaluator
import sys 
from reasoners import LanguageModel, Reasoner, SearchAlgorithm

from gsm8k_rm.evaluation.world_model import GSM8kWorldModel, GSM8kState, GSM8kAction, GSM8kPromptDict
from gsm8k_rm.evaluation.search_config_with_prm import GSM8kConfigWithPRM
import gsm8k_rm.evaluation.gsm8k_utils as utils
from gsm8k_rm.process_reward_model import TrainedProcessRewardModel, CoTProcessRewardModel, FewshotCoTProcessRewardModel, InterleavedCoTProcessRewardModel
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import re
import wandb
import gc
import os
import json
from nltk.tokenize import sent_tokenize
from vllm.distributed.parallel_state import destroy_model_parallel
from gsm8k_rm.evaluation.gsm8k_utils import format_question_with_chat_template
from prompts.llm_as_judge_every_step_prompts import PROMPT_ICL, PROMPT_CHAT_TEMPLATE
## disable all warnings
import warnings
warnings.filterwarnings("ignore")

ORACLE_PRM = True

def extract_steps_from_solution_for_prm(solution) -> list[str]:
    steps = re.split(r'Step \d+:', solution)
    new_steps = []
    for step in steps:
        step = re.sub(r'Step \d+:', '', step).strip()
        step = step.replace("Let's think step-by-step.", "").replace("Let's think step by step.", "")
        ## handle Minerva answer format
        if 'Final Answer: The final answer is' in step:
            # First check if there are newlines before "Final Answer"
            if '\nFinal Answer: The final answer is' in step:
                # Replace any number of newlines with a single space
                step = re.sub(r'\n+Final Answer: The final answer is', ' The answer is', step)
            else:
                # No newlines, just do direct replacement
                step = step.replace('Final Answer: The final answer is', 'The answer is')
                
        if not step.strip():
            continue
        
        new_steps.append(step)
            
    return new_steps

def aggregate_step_score(step_scores, method):
    # mean, min, max, sum
    if method == 'mean':
        return np.mean(step_scores)
    elif method == 'min':
        return np.min(step_scores)
    elif method == 'max':
        return np.max(step_scores)
    elif method == 'sum':
        return np.sum(step_scores)
    elif method == 'longest_correct_prefix':
        for i, score in enumerate(step_scores):
            if score == 0:
                return i / len(step_scores)
        return len(step_scores)
    else:
        raise ValueError(f'unknown aggregate method: {method}')

def free_vllm_memory(model):
    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    import ray
    ray.shutdown()

def bon_gsm8k(base_model: LanguageModel,
              prompt: GSM8kPromptDict,
              prm_path: str,
              prm_type: Literal['segregated', 'interleaved', 'discriminative'],
              use_baseline_prm: bool = False,
              batch_size: int = 2,
              temperature: float = 0.8,
              n_samples: int = 10,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              data_split: str = 'test',
              aggregate_method: str = 'mean',
              prm_bsz: int = 40,
              prm_n: int = 1,
              prm_temperature: float = 0.0,
              seed: int = 0,
              output_path: str = None):

    gsm8k_config = GSM8kConfigWithPRM(base_model=None, prm=None) # we only need this for update_example() to get the prompt
    gsm8k_config.update_example(None, prompt)
    prompt_examples = gsm8k_config.prompt_examples
    sampling_params = SamplingParams(temperature=temperature, 
                                     max_tokens=512, 
                                     n=n_samples, 
                                     stop=["\n\n", "Q:"],
                                     seed=seed)

    evaluator = GSM8KEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=utils.retrieve_answer_from_dataset,
                               init_prompt=prompt,
                               sample_prompt_type="rap",
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm,
                               data_split=data_split
                               )
    
    ## check if the solutions are already cached
    solutions = []
    
    tokenizer = base_model.get_tokenizer()
    model_name = tokenizer.name_or_path.split("/")[-1]
    cache_path = f'outputs/cached_solutions/gsm8k/{model_name}/bon_gsm8k_temp_{temperature}_n_samples_{n_samples}_samples_{data_split}.json'
    
    if os.path.exists(cache_path):
        print('>>>>> Loading cached solutions!')
        with open(cache_path) as f:
            solutions = json.load(f)
    else:
        for i in tqdm(range(0, len(evaluator.full_dataset), batch_size), desc='Sampling solutions'):
            batch = evaluator.full_dataset.select(range(i, min(i+batch_size, len(evaluator.full_dataset))))
            batch_questions = [example['question'] for example in batch]
            batch_prompts = [format_question_with_chat_template(question=q, instruction=prompt['instruction'], examples=prompt['interactive_examples'], tokenizer=tokenizer) for q in batch_questions]
                        
            outputs = base_model.generate(batch_prompts, sampling_params)
            
            for example, output in zip(batch, outputs):
                sampled_solutions = [o.text for o in output.outputs]
                solutions.append({
                    'question': example['question'],
                    'solutions': sampled_solutions,
                    'answer': example['answer']
                })
                
        ### save solutions to a file 
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(solutions, f)
        

    if ORACLE_PRM:
        #### consider a problem solved if any of the solutions is correct
        n_correct = 0
        n_total = 0
        
        print(">>>> Oracle PRM...")
        for solution in solutions:
            gold_answer = evaluator.answer_extractor(solution['answer'])
            if any(evaluator.eval_output(gold_answer, evaluator.output_extractor(o)) for o in solution['solutions']):
                n_correct += 1
            n_total += 1
            
        print(f"Accuracy: {n_correct / n_total}")
        return n_correct / n_total

    ## delete the llm object to free up memory
    free_vllm_memory(base_model)
    destroy_model_parallel()
    ## load prm
    prm = None
    if prm_path is not None:
        if not use_baseline_prm:
            prm_class = {
                'segregated': CoTProcessRewardModel,
                'interleaved': InterleavedCoTProcessRewardModel,
                'discriminative': TrainedProcessRewardModel
            }[prm_type]

            print(f">>>> Loading {prm_type} PRM...")
            prm = prm_class(model_name_or_path=prm_path,
                          max_length=1024,
                          n=prm_n,
                          temperature=prm_temperature)
        else:
            print('>>>>> Using baseline few-shot PRM!')
            prm = FewshotCoTProcessRewardModel(model_name_or_path=prm_path,
                                               max_length=768,
                                               seed=seed,
                                               prompt_template=PROMPT_CHAT_TEMPLATE)
            
    # Process solutions and compute step scores using PRM
    n_correct = 0
    n_total = 0
    all_outputs = []
    
    ### find a solution that has no steps
    for solution in solutions:
        for sol in solution['solutions']:
            if len(extract_steps_from_solution_for_prm(sol)) == 0:
                import ipdb; ipdb.set_trace()

    
    for solution in tqdm(solutions, desc='Ranking solutions'):
        question = solution['question']
        sampled_solutions = solution['solutions']
        answer = solution['answer']

        if prm is not None:
            ## remove empty solutions
            sampled_solutions = [s for s in sampled_solutions if s.strip()]
            processed_samples = []
            for i in range(0, len(sampled_solutions), prm_bsz):
                # Prepare batch inputs for prm.predict_correctness_batch
                batch_steps = [extract_steps_from_solution_for_prm(sample) for sample in sampled_solutions[i:i+prm_bsz]]
                batch_questions = [question] * len(batch_steps)
                
                ## if any solution has only one step, split using sent_tokenize
                for j, steps in enumerate(batch_steps):
                    if len(steps) == 1:
                        batch_steps[j] = sent_tokenize(steps[0])

                # Call predict_correctness_batch
                batch_results = prm.predict_correctness_batch(batch_questions, batch_steps)
            
                for sample, (_, pred_info) in zip(sampled_solutions[i:i+prm_bsz], batch_results):
                    step_scores = pred_info['step_scores']
                    score = aggregate_step_score(step_scores, aggregate_method)
                    
                    processed_samples.append({
                        'steps': [s for s in sample.split('\n') if s.strip()],
                        'step_scores': step_scores,
                        'step_cots': pred_info['step_cots'],
                        'output_texts': pred_info['output_texts'],
                        'score': score,
                        'solution': sample,

                    })
        else:
            processed_samples = [{'steps': [s for s in sample.split('\n') if s.strip()],
                                  'step_scores': None,
                                  'score': 1.0, # dummy score
                                  'solution': sample} for sample in sampled_solutions]

        
        ## sort samples by total_score
        processed_samples = sorted(processed_samples, key=lambda x: x['score'], reverse=True)
        top_solution = processed_samples[0]['solution'] # this is the top solution
        pred_answer = evaluator.output_extractor(re.split(r'Step \d+:', top_solution)[-1]) # this is the answer of the top solution
        gold_answer = evaluator.answer_extractor(answer)
        is_correct = evaluator.eval_output(gold_answer, pred_answer)

        n_correct += is_correct
        n_total += 1

        all_outputs.append({
            'question': question,
            'answer': gold_answer,
            'pred_answer': pred_answer,
            'correct': is_correct,
            'top_solution': top_solution,
            'scored_solutions': processed_samples
        })

        if n_total % 50 == 0:
            ## print accuracy so far
            print(f'[{n_total}/{len(solutions)}] Accuracy: {n_correct / n_total}')


    accuracy = n_correct / n_total
    print(f'Final Accuracy: {accuracy}')

    # Log the results
    wandb.log({
        "accuracy": n_correct / n_total,
        "n_correct": n_correct,
        "n_total": n_total,
    })

    # Close wandb run
    wandb.finish()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_outputs, f)

    return accuracy



if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(base_lm: Literal['hf', 'vllm'] = 'hf',
             hf_path: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct',
             batch_size: int = 1,
             disable_log: bool = False,
             disable_tqdm: bool = False,
             data_split: str = 'test',
             n_samples: int = 10,
             temperature: float = 0.8,
             prompt: str = 'examples/mcts_gsm8k/prompts/prompt_pool.json',
             prm_path: str = None,
             prm_type: Literal['segregated', 'interleaved', 'discriminative'] = 'segregated',
             use_baseline_prm: bool = False,
             aggregate_method: str = 'mean',
             prm_bsz: int = 40,
             prm_n: int = 1,
             prm_temperature: float = 0.0,
             output_path: str = None,
             **kwargs):
        
        with open(prompt) as f:
            prompt = json.load(f)
      
        seed = int(os.environ.get("SEED", 0))
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        base_model = LLM(hf_path, 
                          tensor_parallel_size= 1,
                          seed=seed,
                          enable_prefix_caching=True,
                          gpu_memory_utilization=0.90,
                          max_model_len=8192)
 
        wandb.init(project="self-taught-prm", config={
            "n_samples": n_samples,
            "temperature": temperature,
            "aggregate_method": aggregate_method,
            "prm_path": prm_path,
            "use_baseline_prm": use_baseline_prm,
            "data_split": data_split,
            "model": hf_path if base_lm == 'hf' or base_lm == 'vllm' else base_lm,
        })

        bon_gsm8k(base_model=base_model,
                  prm_path=prm_path,
                  prm_type=prm_type,
                  use_baseline_prm=use_baseline_prm,
                  prompt=prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  data_split=data_split,
                  temperature=temperature,
                  aggregate_method=aggregate_method,
                  seed=seed,
                  n_samples=n_samples,
                  prm_bsz=prm_bsz,
                  prm_n=prm_n,
                  prm_temperature=prm_temperature,
                  output_path=output_path)


    fire.Fire(main)
