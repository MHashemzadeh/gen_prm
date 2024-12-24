from typing import Type, Callable, Optional, Literal
import numpy as np
import sys
import os

# Add the parent of the parent directory to the Python path nvbvb
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from reasoners.benchmark import GSM8KEvaluator
from reasoners.benchmark import GSM8KEvaluator
import sys 
from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation

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

## disable all warnings
import warnings
warnings.filterwarnings("ignore")

ORACLE_PRM = False
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device_1 = torch.device("cuda:0")
device_2 = torch.device("cuda:1")


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

def free_vllm_memory(model, verifier=False):
    
    if not verifier:    del model.llm_engine.model_executor
    del model
    gc.collect()
    torch.cuda.empty_cache()
    import ray
    ray.shutdown()



class bon_gsm8k:
    def __init__(self,
            # base_model: LanguageModel,
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
            output_path: str = None,
            hf_path: str = None):

        
        self.gsm8k_config = GSM8kConfigWithPRM(base_model=None, prm=None) # we only need this for update_example() to get the prompt
        self.gsm8k_config.update_example(None, prompt)
        self.prompt_examples = self.gsm8k_config.prompt_examples
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=512, 
                                        n=n_samples, 
                                        stop=["\n\n", "Question:"],
                                        seed=seed)

        self.evaluator = GSM8KEvaluator(output_extractor=utils.retrieve_answer,
                                answer_extractor=utils.retrieve_answer_from_dataset,
                                init_prompt=prompt,
                                sample_prompt_type="rap",
                                disable_log=disable_log,
                                disable_tqdm=disable_tqdm,
                                data_split=data_split
                                )
        self.temperature = temperature
        self.n_samples = n_samples
        self.data_split = data_split
        # self.base_model = base_model
        self.batch_size = batch_size
        self.seed = seed
        self.prm_path = prm_path
        self.use_baseline_prm = use_baseline_prm
        self.prm_temperature = prm_temperature
        self.prm_type = prm_type
        self.prm_n = prm_n
        self.prm_bsz = prm_bsz
        self.aggregate_method = aggregate_method
        self.output_path = output_path
        self.hf_path=hf_path
        self.total_number_corect = 0
    
    
    ## setup PRM
    def setup_prm(self):
        
        self.prm = None
        if self.prm_path is not None:
            if not self.use_baseline_prm:
                self.prm_class = {
                    'segregated': CoTProcessRewardModel,
                    'interleaved': InterleavedCoTProcessRewardModel,
                    'discriminative': TrainedProcessRewardModel
                }[self.prm_type]

                print(f">>>> Loading {self.prm_type} PRM...")
                self.prm = self.prm_class(model_name_or_path=self.prm_path,
                            max_length=2048,
                            n=self.prm_n,
                            temperature=self.prm_temperature, device=device_2)
            else:
                print('>>>>> Using baseline few-shot PRM!')
                self.prm = FewshotCoTProcessRewardModel(model_name_or_path=self.prm_path,
                                        seed=self.seed)
                

    ## check if the solutions are already cached
    def generate_first_answer(self):
        solutions = []
        
        cache_path = f'outputs/cached_solutions/gsm8k/bon_gsm8k_temp_{self.temperature}_n_samples_{self.n_samples}_samples_{self.data_split}.json'
        if os.path.exists(cache_path):
            print('>>>>> Loading cached solutions!')
            with open(cache_path) as f:
                solutions = json.load(f)
        else:
            ## load base model
            base_model = LLM(model=self.hf_path, 
                          tensor_parallel_size= 1,
                          seed=self.seed,
                          enable_prefix_caching=True,
                          dtype=torch.float16, #"bfloat16", "float16"
                          trust_remote_code=True,
                    )

            for i in tqdm(range(0, len(self.evaluator.full_dataset), self.batch_size), desc='Sampling solutions'):
                batch = self.evaluator.full_dataset.select(range(i, min(i+self.batch_size, len(self.evaluator.full_dataset))))
                batch_questions = [example['question'] for example in batch]
                batch_prompts = [self.prompt_examples + self.gsm8k_config.prompt['question_prefix'].format(question=q) for q in batch_questions]
                
                outputs = base_model.generate(batch_prompts, self.sampling_params)
                
                for example, output in zip(batch, outputs):
                    sampled_solutions = [o.text for o in output.outputs]
                    solutions.append({
                        'question': example['question'],
                    'solutions': sampled_solutions,
                    'answer': example['answer']
                })
                    
                ### save solutions to a file 
                os.makedirs('outputs/cached_solutions/gsm8k', exist_ok=True)
                with open(f'outputs/cached_solutions/gsm8k/bon_gsm8k_temp_{self.temperature}_n_samples_{self.n_samples}_samples_{self.data_split}.json', 'w') as f:
                    json.dump(solutions, f)

            # Clean up after inference
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reset CUDA device to fully clear memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()  # Wait for all streams on the current device
        
        ## delete the llm object to free up memory
        # free_vllm_memory(self.base_model)
        return solutions[:10]

    
    def generate_answer(self, itr, prompt_itr):
        ## check if the solutions are already cached
        solutions = []
        
        cache_path = f'outputs/cached_solutions/gsm8k/bon_gsm8k_temp_{self.temperature}_n_samples_{self.n_samples}_samples_{self.data_split}_iteration_{itr+1}.json'
        if False and os.path.exists(cache_path):
            print('>>>>> Loading cached solutions!')
            with open(cache_path) as f:
                solutions = json.load(f)
        else:
            ## load base model
            base_model = LLM(model=self.hf_path, 
                          tensor_parallel_size= 1,
                          seed=self.seed,
                          enable_prefix_caching=True,
                          dtype=torch.float16, #"bfloat16", "float16"
                          trust_remote_code=True,
                    )
            
            previous_path=f'{self.output_path}/all_outputs_gsm8k_temp_{self.temperature}_n_samples_{self.n_samples}_samples_{self.data_split}_iteration_{itr}.json'
            data = self.evaluator.load_reponse(previous_path)           
            # solutions = [
            #             {
            #                 "question": example["question"],
            #                 "solutions": [example["top_solution"]],  
            #                 "answer": example["answer"]
            #             }
            #             for example in data if example["correct"]
            #         ]
            self.total_number_corect = sum(1 for example in data if example["correct"])
            filtered_data = [example for example in data if not example['correct']]
            
            if filtered_data:
                for i in tqdm(range(0, len(filtered_data), self.batch_size), desc='Sampling solutions'):
                    # all_outputs.append({
                    #                 'question': question,
                    #                 'answer': gold_answer,
                    #                 'pred_answer': pred_answer,
                    #                 'correct': is_correct,
                    #                 'top_solution': top_solution,
                    #                 'scored_solutions': processed_samples
                    #             })
                    

                    batch = filtered_data[i:min(i + self.batch_size, len(filtered_data))]
                    # solutions = [
                    #     {
                    #         "question": example["question"],
                    #         "solutions": example["top_solution"],  
                    #         "answer": example["answer"]
                    #     }
                    #     for example in batch if example["correct"]
                    # ]

                    # filtered_batch = [example for example in batch if not example['correct']]


                    batch_questions = [example['question'] for example in batch]
                    batch_pred_answer = [example['pred_answer'] for example in batch]
                    batch_score = [example['scored_solutions'][0]['score'] for example in batch]
                    batch_step_scores = [example['scored_solutions'][0]['step_scores'] for example in batch]
                    batch_steps = [example['scored_solutions'][0]['steps'] for example in batch]
                    batch_correct = [example['correct'] for example in batch]
                    batch_top_solution = [example['top_solution'] for example in batch]


                    batch_prompts = []

                    for question, pred_answer, step, step_scores, top_solution in zip(batch_questions, batch_pred_answer, batch_steps, batch_step_scores, batch_top_solution):
                        # Combine `question`, `pred_answer`, `step_cots`, and `step_scores` with `prompt_examples`.
                        
                        prompt = (
                            # f"{self.prompt_examples}"
                            f"Here is the previous generated response which is not correct!"
                            f"Question: {question}\n"
                            f"Predicted Answer: {pred_answer}\n"
                            # f"Previous Solution: {top_solution}\n"
                            "Here is the Step-by-step reasoning and corresponding scores:\n" +
                            "".join(f"Step {i + 1}: {cot} (Score: {score})\n" for i, (cot, score) in enumerate(zip(step, step_scores))) +
                            "\nNote: A score closer to 1 indicates higher correctness.\n"
                            "Using these scores, change the steps with low scores. "
                            "Then, regenerate the entire response to achieve the correct answer. You should follow the step step reasoning and then asnwer like examples:"
                            f"{self.prompt_examples + self.gsm8k_config.prompt['question_prefix'].format(question=question)}"
                        )
                        
                        
                        batch_prompts.append(prompt)

                    # batch_prompts = [prompt_examples + gsm8k_config.prompt['question_prefix'].format(question=q) for q in batch_questions]
                    
                    outputs = base_model.generate(batch_prompts, self.sampling_params)
                    
                    for example, output in zip(batch, outputs):
                        sampled_solutions = [o.text for o in output.outputs]
                        solutions.append({
                            'question': example['question'],
                        'solutions': sampled_solutions,
                        'answer': example['answer']
                    })
                        
                    ### save solutions to a file 
                    os.makedirs('outputs/cached_solutions/gsm8k', exist_ok=True)
                    with open(cache_path, 'w') as f:
                        json.dump(solutions, f)
        
            # Clean up after inference
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reset CUDA device to fully clear memory
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()  # Wait for all streams on the current device

        return solutions


    def calculate_prm_score(self, solutions, itr):
        # Process solutions and compute step scores using PRM
        n_correct = self.total_number_corect #0
        n_total = self.total_number_corect 
        all_outputs = []
        
        

        if ORACLE_PRM:
            #### consider a problem solved if any of the solutions is correct
            # n_correct = self.total_number_corect #0
            # n_total = self.total_number_corect #0
            
            print(">>>> Oracle PRM...")
            for solution in solutions:
                gold_answer = self.evaluator.answer_extractor(solution['answer'])
                if any(self.evaluator.eval_output(gold_answer, self.evaluator.output_extractor(re.split(r'Step \d+:', o)[-1])) for o in solution['solutions']):
                    n_correct += 1
                n_total += 1
                
            print(f"Accuracy: {n_correct / n_total}")
            return n_correct / n_total, None, None
        
        ## load prm model
        print(f">>>> Loading {self.prm_type} PRM MODEL ...")
        self.prm.load_llm()

        for solution in tqdm(solutions, desc='Ranking solutions'):
            question = solution['question']
            sampled_solutions = solution['solutions']
            answer = solution['answer']

            if self.prm is not None:
                ## remove empty solutions
                sampled_solutions = [s for s in sampled_solutions if s.strip()]
                processed_samples = []
                for i in range(0, len(sampled_solutions), self.prm_bsz):
                    # Prepare batch inputs for prm.predict_correctness_batch
                    batch_steps = [[s.strip() for s in re.split(r'Step \d+:', sample) if s.strip()] for sample in sampled_solutions[i:i+self.prm_bsz]] # list of list of steps
                    batch_questions = [question] * len(batch_steps)

                    # Call predict_correctness_batch
                    batch_results = self.prm.predict_correctness_batch(batch_questions, batch_steps)
                
                    for sample, (_, pred_info) in zip(sampled_solutions[i:i+self.prm_bsz], batch_results):
                        step_scores = pred_info['step_scores']
                        score = aggregate_step_score(step_scores, self.aggregate_method)
                        
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
            pred_answer = self.evaluator.output_extractor(re.split(r'Step \d+:', top_solution)[-1]) # this is the answer of the top solution
            gold_answer = self.evaluator.answer_extractor(answer)
            is_correct = self.evaluator.eval_output(gold_answer, pred_answer)

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


        # Clean up after inference
        del self.prm.llm
        gc.collect()
        torch.cuda.empty_cache()
        # Reset CUDA device to fully clear memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()  # Wait for all streams on the current device
        ####

        accuracy = n_correct / n_total
        print(f'Accuracy: {accuracy}')

        # Log the results
        wandb.log({
            "accuracy": n_correct / n_total,
            "n_correct": n_correct,
            "n_total": n_total,
        })

        # # Close wandb run
        # wandb.finish()
        ## free memory
        # free_vllm_memory(self.prm, True)

        if self.output_path is not None:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            output_path_all_outputs = f'{self.output_path}/all_outputs_gsm8k_temp_{self.temperature}_n_samples_{self.n_samples}_samples_{self.data_split}_iteration_{itr}.json'
            with open(output_path_all_outputs, 'w') as f:
                json.dump(all_outputs, f)

        return accuracy, all_outputs, output_path_all_outputs


    def iterative_correctness(self, num_itr):
        
        self.setup_prm()
        solutions = self.generate_first_answer()
        # self.load_prm()
        accuracies = []
        for i in range(num_itr):

            accuracy, all_outputs, output_path_all_outputs = self.calculate_prm_score(solutions, i)
 
            accuracies.append(accuracy)
            prompt_itr = 'Here is the previous thoughts and the level of' + \
                ' their correctness as scores. The highest score is one and lowest is zero. ' + \
                'It means if the score is closer to one it is more correct. ' + \
                'Use the score and then correct the steps with low score in order to find the correct answer to the question.'
            
            
            # free_vllm_memory(self.prm, True)

            solutions = self.generate_answer(i, prompt_itr)

        
        print(accuracies)  
        # Close wandb run
        wandb.finish()






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
             hf_path: str ='/home/mila/m/maryam.hashemzadeh/scratch/cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a', #'google/gemma-2b-it', #'meta-llama/Meta-Llama-3-8B-Instruct', #'meta-llama/Llama-2-13b-hf',
             batch_size: int = 10,
             disable_log: bool = False,
             disable_tqdm: bool = False,
             data_split: str = 'test',
             n_samples: int = 10,
             temperature: float = 0.8,
             prompt: str = 'mcts-distill/examples/mcts_gsm8k/prompts/prompt_pool_3shot.json', #'examples/mcts_gsm8k/prompts/prompt_pool.json',
             prm_path: str ='/home/mila/m/maryam.hashemzadeh/scratch/verifier/chpt/new/experimental-models/gen-p-seg', #'/scratch/cache/huggingface/hub/models--mkhalifa--experimental-models/snapshots/790037d70c2fe988830c69a075cb746cb33238e2/gen-p-seg', #'/home/mila/m/maryam.hashemzadeh/scratch/verifier/chpt/experimental-models/gen-p-seg',
             prm_type: Literal['segregated', 'interleaved', 'discriminative'] = 'segregated',
             use_baseline_prm: bool = False,
             aggregate_method: str = 'mean',
             prm_bsz: int = 40,
             prm_n: int = 1,
             prm_temperature: float = 0.0,
             output_path: str = 'outputs/cached_solutions/gsm8k',
             num_iteration: int = 3,
             **kwargs):
        
        with open(prompt) as f:
            prompt = json.load(f)
      
        seed = int(os.environ.get("SEED", 0))
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

        # base_model = LLM(model=hf_path, 
        #                   tensor_parallel_size= 1,
        #                   seed=seed,
        #                   enable_prefix_caching=True,
        #                 #   dtype="bfloat16", #"bfloat16", "float16"
        #                   device = "cuda:0",
        #             )
 
        wandb.init(project="self-taught-prm", config={
            "n_samples": n_samples,
            "temperature": temperature,
            "aggregate_method": aggregate_method,
            "prm_path": prm_path,
            "use_baseline_prm": use_baseline_prm,
            "data_split": data_split,
            "model": hf_path if base_lm == 'hf' or base_lm == 'vllm' else base_lm,
        })

        gsm8k_agent = bon_gsm8k(#base_model=base_model,
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
                  output_path=output_path,
                  hf_path=hf_path)
        
        # gsm8k_agent.load_prm()
        
        accuracy = gsm8k_agent.iterative_correctness(num_itr=num_iteration)

    fire.Fire(main)
