
from typing import TypedDict, Optional
import sys 
from reasoners import SearchConfig, LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import io
import torch
import random
from vllm import LLM, SamplingParams
from utils.prm_dataset import PRMTrajectoryDataset, PRMCoTEvalDataset, PRMCoTInterleavedDataset
from utils.config import Config
from utils.answer_utils import extract_step_cots_and_labels
from prompts.llm_as_judge_every_step_prompts import PROMPT_ICL
import re
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class PromptProcessRewardModel:
    def __init__(self,
                base_model: LanguageModel,
                prm_prompt: dict,
                tokenizer: AutoTokenizer,
                n_actions=4,
                batch_size=1,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                reward_alpha=0.5,
                reward_confidence_default=0.8,
                depth_limit=5,
                force_terminating_on_depth_limit=True,
                force_overall_prompt_on_overall_question=True,
                force_overall_question_on_overall_prompt=True,
                random_reward=False,
                ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prm_prompt = prm_prompt
        self.example = ''
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.force_overall_prompt_on_overall_question = force_overall_prompt_on_overall_question
        self.force_overall_question_on_overall_prompt = force_overall_question_on_overall_prompt
        self.overall_question: Optional[str] = None
        self.prompt_examples = ""
        self.n_shots = 0
        self.random_reward = random_reward
        self.tokenizer = tokenizer

        if random_reward:
            print("Random reward is enabled, the reward will be a random number between 0 and 1")

        assert prm_prompt is not None
        self.prm_prompt = prm_prompt


    def predict_correctness(self, question: str, steps: list[str]) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.prm_prompt["input"])
            f.write(self.prm_prompt["question_prefix"].format(question=question) + "\n")
            
            for idx, step in enumerate(steps):
                if not step.endswith('.'):
                    step += '.'
                f.write(self.prm_prompt["step_prefix"].format(idx=idx + 1) + " " + step + "\n")
            
            f.write(self.prm_prompt["useful_prefix"] + " ")
            model_input = f.getvalue()

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        cor_prob = probs[0]
        return cor_prob

    def predict_correctness_batch(self, questions: list[str], steps: list[list[str]]) -> tuple[list[float], list[dict]]:
        # Prepare a list to hold all model inputs for batching
        model_inputs = []
        # Create model input for each question and its corresponding steps
        for question, step_list in zip(questions, steps):
            with io.StringIO() as f:
                if 'template' in self.prm_prompt:
                    #### chat template mode
                    solution = ' '.join(step_list)
                    f.write(self.prm_prompt["template"].format(question=question, solution=solution))
                
                else:
                    f.write(self.prm_prompt["input"])
                    f.write(self.prm_prompt["question_prefix"].format(question=question) + "\n")
                    for idx, step in enumerate(step_list):
                        if not step.strip().endswith('.'):
                            step = step.strip() + '.'
                        if self.prm_prompt["step_prefix"].strip():
                            f.write(self.prm_prompt["step_prefix"].format(idx=idx + 1) + " ")
                        f.write(step + "\n")
                    f.write(self.prm_prompt["useful_prefix"] + "")
                
                model_input = f.getvalue()
                model_inputs.append(model_input)
        
        #import ipdb; ipdb.set_trace()
        # Call the base model's get_next_token_logits with the batched input
        logits_batch = self.base_model.get_next_token_logits(model_inputs, ["Yes", "No"])
        assert len(logits_batch) == len(questions)
        
        # Process the logits to compute probabilities
        batch_cor_probs = []

        for logits in logits_batch:
            probs = np.exp(logits) / np.sum(np.exp(logits))
            cor_prob = probs[0]
            batch_cor_probs.append(cor_prob)

        # Return the batched correctness probabilities and logit information
        return batch_cor_probs
    
    def predict_correctness_cot_batch(self, questions: list[str], steps: list[list[str]], step_labels: list[int]) -> tuple[list[float], list[dict]]:
        # Prepare a list to hold all model inputs for batching
        model_inputs = []
        # Create model input for each question and its corresponding steps
        for question, step_list in zip(questions, steps):
            with io.StringIO() as f:
                assert 'template' in self.prm_prompt
                    #### chat template mode
                solution = "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(step_list))
                f.write(self.prm_prompt["template"].format(question=question, solution=solution))
                
                model_input = f.getvalue()
                model_inputs.append(model_input)
        
        # Call the base model's get_next_token_logits with the batched input
        sampling_params = SamplingParams(
            max_tokens=256,
            stop=[" Yes", " No"],
            logprobs=100,
        )
        output = self.base_model.generate(model_inputs, sampling_params)
        outputs = [o.outputs[0].text for o in output]
        
        ### logprobs
        logprobs = [o.outputs[0].logprobs[-1] for o in output]
        yes_token = " Yes"
        no_token = " No"

        yes_token_id = self.tokenizer.encode(yes_token, add_special_tokens=False)[-1]
        no_token_id = self.tokenizer.encode(no_token, add_special_tokens=False)[-1]
        
        batch_cor_probs = []
        for lp, model_input, cot, gt_label in zip(logprobs, model_inputs, outputs, step_labels):
            if yes_token_id not in lp or no_token_id not in lp:
                print("no yes or no token in the logprobs...")
                ## set to random
                cor_prob = 0
                batch_cor_probs.append(cor_prob)
                continue

            yes_logprob = lp[yes_token_id].logprob
            no_logprob = lp[no_token_id].logprob
            cor_prob = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
            pred_label = 1 if cor_prob > 0.5 else 0
            #if pred_label != gt_label:
            #    import ipdb; ipdb.set_trace()

            batch_cor_probs.append(cor_prob)
        
        assert len(batch_cor_probs) == len(questions)
        # Return the batched correctness probabilities and logit information
        return batch_cor_probs


class TrainedProcessRewardModel:
    def __init__(self,
                model_name_or_path: str,
                step_sep: str = ' ки',
                pos_label_step_token: str = '+',
                neg_label_step_token: str = '-',
                random_reward: bool = False,
                max_length: int = 1024,
                device: str = 'cuda',
                ) -> None:
        super().__init__()
        
        print("Loading PRM model from {}".format(model_name_or_path))
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.step_sep_id = self.tokenizer.encode(step_sep, add_special_tokens=False)[-1]
        self.pos_step_id = self.tokenizer.encode(pos_label_step_token, add_special_tokens=False)[-1]
        self.neg_step_id = self.tokenizer.encode(neg_label_step_token, add_special_tokens=False)[-1]
        self.random_reward = random_reward
        self.max_length = max_length


    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        inputs = self.process_example(question, prefix_steps)
        if inputs['input_ids'][-1] != self.step_sep_id:
            print("Warning: step separator not found in the input ids, adding it...")
            inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])

        input_ids = inputs['input_ids'].unsqueeze(0)  # Add batch dimension
        attention_mask = inputs['attention_mask'].unsqueeze(0)

        # Move tensors to the same device as the model
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        candidate_tokens = [self.pos_step_id, self.neg_step_id]

        # Get model outputs
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len
            step_scores = scores[input_ids == self.step_sep_id] # 1 x 
            step_scores = step_scores.cpu().tolist()

        full_prefix_score = step_scores[-1]
        
        info = {
            'full_prefix_score': full_prefix_score,
            'step_scores': step_scores,
        }

        return full_prefix_score, info

    def predict_correctness_batch(self, questions: list[str], prefix_steps_list: list[list[str]]) -> list[tuple[float, dict]]:
        # Tokenize all inputs at once
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_list):
            inputs = self.process_example(question, prefix_steps)
            if inputs['input_ids'][-1] != self.step_sep_id:
                inputs['input_ids'] = torch.cat([inputs['input_ids'], torch.tensor([self.step_sep_id])])
                inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.tensor([1])])
            batch_inputs.append(inputs)

        # Pad sequences to max length in batch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [inputs['input_ids'] for inputs in batch_inputs], 
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        ).to(self.model.device)
        
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [inputs['attention_mask'] for inputs in batch_inputs],
            batch_first=True,
            padding_value=0
        ).to(self.model.device)

        candidate_tokens = [self.pos_step_id, self.neg_step_id]

        # Get model outputs for entire batch at once
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask=attention_mask).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len

        # Process results for each example
        results = []
        for i, prefix_steps in enumerate(prefix_steps_list):
            step_scores = scores[i][input_ids[i] == self.step_sep_id].cpu().tolist()
            if len(step_scores) != len(prefix_steps): # TODO fix this 
                print("Warning: step scores and prefix steps are not of the same length. This is likely due to a very long chain.")
            full_prefix_score = step_scores[-1]
            
            info = {
                'full_prefix_score': full_prefix_score,
                'step_scores': step_scores,
            }

            results.append((full_prefix_score, info))

        return results

    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization
        example = {
            'question': question,
            'steps_with_labels': [{'step': step, 'label': '+'} for step in prefix_steps], # placeholder labels
            'solution_label': -1 # placeholder label
        }

        # Call tokenize_example from prm_dataset.py
        tokenized_example = PRMTrajectoryDataset.tokenize_example(
            example, 
            self.tokenizer, 
            self.step_sep_id, 
            self.pos_step_id, 
            self.neg_step_id, 
            self.max_length,
            config={},
            split='test'
        )

        # Extract the required fields
        input_ids = tokenized_example['input_ids']
        attention_mask = tokenized_example['attention_mask']
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    
class CoTProcessRewardModel:
    def __init__(self,
                model_name_or_path: str,
                max_length: int = 1024,
                device: str = 'cuda',
                n: int = 1,
                temperature: float = 0.0,
                seed: int = 0,
                ) -> None:
        super().__init__()
        
        print("Loading PRM model from {}".format(model_name_or_path))

        
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        # self.llm = LLM(model_name_or_path,
        #             tensor_parallel_size=1,
        #             seed=seed,
        #             gpu_memory_utilization=0.50,
        #             max_num_batched_tokens=max_length,
        #             max_model_len=max_length,
        #             max_logprobs=1000,
        #             # dtype="bfloat16",
        #             # device="cuda:1",
        #             )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.seed = seed
        self.max_length = max_length
        self.model_name_or_path = model_name_or_path
        
        self.sampling_params = SamplingParams(
            max_tokens=self.max_length,
            seed=self.seed,
            temperature=temperature,
            n=n,
            logprobs=100,
            frequency_penalty=.1,
        )
       
       ### create a fake config 
        self.config = {
            'task': 'gsm8k',
            'data_dir': '',
            'debug': True,
            'max_length': self.max_length,
        }

        ### convert to a namespace object we can access by config.
        dataset_config = Config(self.config)

        ### create a dataset objective
        self.dataset_obj = PRMCoTEvalDataset(
            examples=[],
            tokenizer=self.tokenizer,
            process_data=False,
            config=dataset_config,
            split='test'
        )

        yes_token = " Yes"
        no_token = " No"
        self.yes_token_id = self.tokenizer.encode(yes_token, add_special_tokens=False)[-1]
        self.no_token_id = self.tokenizer.encode(no_token, add_special_tokens=False)[-1]

    def load_llm(self):
        self.llm = LLM(self.model_name_or_path,
                    tensor_parallel_size=1,
                    seed=self.seed,
                    gpu_memory_utilization=0.50,
                    max_num_batched_tokens=self.max_length,
                    max_model_len=self.max_length,
                    max_logprobs=1000,
                    dtype=torch.float16,
                    trust_remote_code=True,
                    # device="cuda:1",
                    )

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        input_text = self.process_example(question, prefix_steps)
        
        output = self.generation_pipeline([input_text])
        output = output[0][0]['generated_text']
        
        step_cots, step_labels = extract_step_cots_and_labels(output)
        ## score would be avg of step_labels
        score = sum(step_labels) / len(step_labels)

        info = {
            'step_cots': step_cots,
            'step_labels': step_labels,
        }

        return score, info
    
    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process all examples in the batch
        input_texts = [self.process_example(question, prefix_steps) 
                       for question, prefix_steps in zip(questions, prefix_steps_batch)]
                
        n = self.sampling_params.n
        # Generate outputs for the entire batch
        outputs = self.llm.generate(input_texts, self.sampling_params, use_tqdm=False)

        results = []
        for i, input_text in enumerate(input_texts):
            # Process all generations for this input
            all_step_scores = []
            all_step_cots = []
            all_step_labels = []
            all_outputs = []
            
            for output in outputs[i].outputs:
                solution_output = output.text
                solution_logprobs = output.logprobs
                
                step_scores = self.get_step_scores_from_logprobs(solution_logprobs, solution_output, prefix_steps_batch[i])
                step_cots, step_labels = extract_step_cots_and_labels(solution_output)
                
                all_step_scores.append(step_scores)
                all_step_cots.append(step_cots)
                all_step_labels.append(step_labels)
                all_outputs.append(solution_output)
            
            # Calculate average scores, ignoring NaN values
            avg_step_scores = np.nanmean(all_step_scores, axis=0)
            # If all scores for a step are NaN, use neutral score
            avg_step_scores = np.nan_to_num(avg_step_scores, nan=0.5)
            full_prefix_score = avg_step_scores[-1]

            avg_step_scores = avg_step_scores.tolist()

            info = {
                'step_cots': all_step_cots,
                'step_labels': all_step_labels, 
                'step_scores': avg_step_scores,
                'input_text': input_text,
                'output_texts': all_outputs,
            }
            
            results.append((full_prefix_score, info))
        
        return results
    
    def get_step_scores_from_logprobs(self, solution_logprobs, solution_output, prefix_steps):
        """Extract step scores from logprobs by comparing Yes/No token probabilities."""

        step_scores = []
        yes_no_positions = []
        
        # First pass: identify all Yes/No decision points and their positions
        for pos, lp_info in enumerate(solution_logprobs):
            top_token_id = next(k for k, v in lp_info.items() if v.rank == 1)
            if top_token_id in (self.yes_token_id, self.no_token_id):
                yes_no_positions.append(pos)
                
        if not yes_no_positions:
            print(f"Warning: No Yes/No decisions found in output {solution_output}")
            return [0.5] * len(prefix_steps)  # Return neutral scores
            
        #if len(yes_no_positions) != len(prefix_steps):
        #    print(f"Warning: Mismatch between number of Yes/No decisions ({len(yes_no_positions)}) "
        #        f"and steps ({len(prefix_steps)})")
            
        # Second pass: compute scores and align with steps
        for pos in yes_no_positions:
            lp_info = solution_logprobs[pos]
            try:
                yes_logprob = next(lp.logprob for id, lp in lp_info.items() 
                                if id == self.yes_token_id)
                no_logprob = next(lp.logprob for id, lp in lp_info.items() 
                                if id == self.no_token_id)
                
                score = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
                step_scores.append(score)
                
            except StopIteration:
                #print(f"Warning: Missing Yes/No logprobs at position {pos}")
                step_scores.append(np.nan)

        # Handle length mismatches more explicitly
        if len(step_scores) < len(prefix_steps):
            #print("Warning: Fewer scores than steps - some steps may not have been evaluated")
            return step_scores + [np.nan] * (len(prefix_steps) - len(step_scores))
        elif len(step_scores) > len(prefix_steps):
            #print("Warning: More scores than steps - some scores may be spurious")
            return step_scores[:len(prefix_steps)]
            
        return step_scores

    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization

        assert len(prefix_steps) > 0, "Prefix steps should not be empty"

        new_prefix_steps = []
        for i, step in enumerate(prefix_steps):
            step = re.sub(r'Step \d+:', '', step).strip()
            step = f'Step {i+1}: {step}'
            new_prefix_steps.append(step)

        solution = '\n'.join(new_prefix_steps)
        # Call tokenize_example from prm_dataset.py
        input_text = self.dataset_obj.format_cot_data(problem=question, solution=solution)

        return input_text
    

class InterleavedCoTProcessRewardModel(CoTProcessRewardModel):
    def __init__(self,
                model_name_or_path: str,
                max_length: int = 1400,
                device: str = 'cuda',
                seed: int = 0,
                ) -> None:
        super().__init__(model_name_or_path, max_length, device, seed)

       ### create a fake config 
        self.config = {
            'data_dir': '',
            'debug': True,
            'max_length': self.max_length,
        }

        ### convert to a namespace object we can access by config.
        dataset_config = Config(self.config)

        ### create a dataset objective
        self.dataset_obj = PRMCoTInterleavedDataset(
            examples=[],
            tokenizer=self.tokenizer,
            process_data=False,
            config=dataset_config,
            split='test'
        )

    def predict_correctness(self, question: str, prefix_steps: list[str], previous_cots: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        input_text = self.process_example(question, prefix_steps, previous_cots)
        
        output = self.generation_pipeline([input_text])
        output = output[0][0]['generated_text']
        
        step_cots, step_labels = extract_step_cots_and_labels(output)
        ## score would be avg of step_labels
        score = sum(step_labels) / len(step_labels)

        info = {
            'step_cots': step_cots,
            'step_labels': step_labels,
        }

        return score, info
    
    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]], previous_cots_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process all examples in the batch
        input_ids_batch = [self.process_example(question, prefix_steps, previous_cots) 
                       for question, prefix_steps, previous_cots in zip(questions, prefix_steps_batch, previous_cots_batch)]
        
        input_ids_batch = [d['input_ids'] for d in input_ids_batch]
        attention_mask_batch = [d['attention_mask'] for d in input_ids_batch]
        
        yes_token = " Yes"
        no_token = " No"
        yes_token_id = self.tokenizer.encode(yes_token, add_special_tokens=False)[-1]
        no_token_id = self.tokenizer.encode(no_token, add_special_tokens=False)[-1]
        
        n = self.sampling_params.n
        # Generate outputs for the entire batch
        outputs = self.llm.generate(input_ids_batch, self.sampling_params)

        results = []
        for i, input_ids in enumerate(input_ids_batch):
            solution_output = outputs[i].outputs[0].text
            solution_logprobs = outputs[i].outputs[0].logprobs
            step_scores = []
            score = None 
            
            for lp_info in solution_logprobs:
                ### find the top rank token and check if it is yes or no
                top_token_id = [k for k, v in lp_info.items() if v.rank == 1][0]
                if top_token_id == yes_token_id or top_token_id == no_token_id:
                    ### find the logprob of Yes and No
                    try:
                        yes_logprob = [lp.logprob for id, lp in lp_info.items() if id == yes_token_id][0]
                        no_logprob = [lp.logprob for id, lp in lp_info.items() if id == no_token_id][0]
                        score = np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))
                        step_scores.append(score)
                    except:
                        import ipdb; ipdb.set_trace()
                        continue

            if score is None:
                print("No step scores found for the solution...")
        
            #### scores are probabilities of yes token or 1 - probability of no token
            step_cots, step_labels = extract_step_cots_and_labels(solution_output)
            
            info = {
                'step_cots': step_cots,
                'step_labels': step_labels,
                'step_scores': step_scores,
                'input_ids': input_ids,
                'output_text': solution_output,
            }
            try:
                results.append(info)
            except:
                import ipdb; ipdb.set_trace()
        
        return results

    def process_example(self, question: str, prefix_steps: list[str], previous_cots: list[str]):
        # Prepare the example for tokenization
        assert len(prefix_steps) == len(previous_cots) + 1
        
        example = {
            'problem': question,
            'solution_steps': prefix_steps,
            'cot_steps': previous_cots,
        }

        model_prompt_ids = self.dataset_obj.format_inst_question(example) # input_ids, attention_mask

        return model_prompt_ids

    
class FewshotCoTProcessRewardModel:
    def __init__(self,
                model_name_or_path: str,
                max_length: int = 1024,
                device: str = 'cuda',
                seed: int = 0,
                n: int = 1,
                temperature: float = 0.7,
                ) -> None:
        super().__init__()
        
        print("Loading PRM model from {}".format(model_name_or_path))
        
        self.llm = LLM(model_name_or_path,
                    tensor_parallel_size=1,
                    seed=seed,
                    gpu_memory_utilization=0.90,
                    max_model_len=16000,
                    enable_prefix_caching=True,
                    )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.sampling_params = SamplingParams(
            max_tokens=self.max_length,
            stop=["\n\n"],
            seed=seed,
            temperature=(0.0 if n == 1 else temperature),
            n=n,
        )
       
        self.seed = seed
        self.max_length = max_length

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        # Tokenize the input
        input_text = self.process_example(question, prefix_steps)
        output = self.generation_pipeline([input_text])
        output = output[0][0]['generated_text']
        
        step_cots, step_labels = extract_step_cots_and_labels(output)
        ## score would be avg of step_labels
        score = sum(step_labels) / len(step_labels)

        info = {
            'step_cots': step_cots,
            'step_labels': step_labels,
        }

        return score, info
    
    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process all examples in the batch
        input_texts = [self.process_example(question, prefix_steps) 
                       for question, prefix_steps in zip(questions, prefix_steps_batch)]
        # Generate outputs for the entire batch
        outputs = self.llm.generate(input_texts, self.sampling_params)
        
        results = []
        for i, input_text in enumerate(input_texts):
            all_outputs = [output.outputs[j].text.strip() for output in outputs for j in range(self.sampling_params.n)]
            
            all_step_cots = []
            all_step_labels = []
            for output in all_outputs[i*self.sampling_params.n:(i+1)*self.sampling_params.n]:
                step_cots, step_labels = extract_step_cots_and_labels(output)
                all_step_cots.extend(step_cots)
                all_step_labels.extend(step_labels)
            
            # Aggregate scores across all samples
            aggregated_step_labels = [sum(sample) for sample in all_step_labels]
            total_samples = len(aggregated_step_labels)
            
            # Calculate aggregated score
            score = sum(aggregated_step_labels) / total_samples if total_samples > 0 else 0
            
            info = {
                'step_cots': all_step_cots,
                'step_labels': all_step_labels,
                'aggregated_step_labels': aggregated_step_labels,
                'score': score,
                'input_text': input_text,
                'output_texts': all_outputs[i*self.sampling_params.n:(i+1)*self.sampling_params.n],
            }
            
            results.append((score, info))
        
        return results

    def process_example(self, question: str, prefix_steps: list[str]):
        # Prepare the example for tokenization
        if not all([step.startswith('Step') for step in prefix_steps]):
            prefix_steps = [f'Step {i+1}: {step.strip()}' for i, step in enumerate(prefix_steps)]
        
        solution = '\n'.join(prefix_steps)
        # Call tokenize_example from prm_dataset.py
        input_text = PROMPT_ICL.format(problem=question, solution=solution)

        return input_text
       
    

class MathShepherdPRM:
    def __init__(self,
                device: str = 'cuda',
                ) -> None:
        
        good_token = '+'
        bad_token = '-'
        step_tag = 'ки'

        self.tokenizer = AutoTokenizer.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm')
        self.candidate_tokens = self.tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [648, 387]


        print("Loading PRM model from peiyi9979/math-shepherd-mistral-7b-prm")
        self.model = AutoModelForCausalLM.from_pretrained('peiyi9979/math-shepherd-mistral-7b-prm').eval()
        self.device = device
        self.model.to(self.device)
        
        self.step_tag_id = self.tokenizer.encode(f"{step_tag}")[-1] # 12902
        self.step_tag = step_tag

    def predict_correctness(self, question: str, prefix_steps: list[str]) -> tuple[float, dict]:
        output = ""
        for i, step in enumerate(prefix_steps, 1):
            output += f"Step {i}: {step} {self.step_tag}\n"
        
        output = output.strip()
        input_for_prm = f"{question} {output}"
        input_ids = torch.tensor([self.tokenizer.encode(input_for_prm)]).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0] # 1 x seq_len
            step_scores = scores[input_ids == self.step_tag_id] # 1 x 

        step_scores = step_scores.cpu().tolist()

        if len(step_scores) != len(prefix_steps):
            print("warning: something probably wrong happened with tokenization that add/removed a step tag")

        prefix_score = step_scores[-1]
        
        info = {
            'full_prefix_score': prefix_score,
            'step_scores': step_scores,
        }

        return prefix_score, info


    def predict_correctness_batch(self, questions: list[str], prefix_steps_batch: list[list[str]]) -> list[tuple[float, dict]]:
        # Process each example into formatted input string
        batch_inputs = []
        for question, prefix_steps in zip(questions, prefix_steps_batch):
            output = ""
            for i, step in enumerate(prefix_steps, 1):
                output += f"Step {i}: {step} {self.step_tag}\n"
            output = output.strip()
            batch_inputs.append(f"{question} {output}")

        # Tokenize all inputs
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        input_ids = self.tokenizer(batch_inputs, padding=True, return_tensors="pt").input_ids.to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]  # batch_size x seq_len
        
        # Extract scores for each step tag
        results = []
        for i, prefix_steps in enumerate(prefix_steps_batch):
            step_mask = (input_ids[i] == self.step_tag_id)
            step_scores = scores[i][step_mask].cpu().tolist()
            
            if len(step_scores) != len(prefix_steps):
                print("warning: something probably wrong happened with tokenization that add/removed a step tag")
            
            prefix_score = step_scores[-1] # last step score is the full prefix score
            
            info = {
                'full_prefix_score': prefix_score,
                'step_scores': step_scores,
            }
            
            results.append((prefix_score, info))
            
        return results
