import io
import re
from typing import TypedDict, Optional
import numpy as np
import random

from world_model import GSM8kState, GSM8kAction, GSM8kPromptDict
from reasoners import SearchConfig, LanguageModel
from collections import defaultdict
import sys 
sys.path.append('..')
from prm.process_reward_model import TrainedProcessRewardModel
from vllm import SamplingParams

class GSM8kConfigWithPRM(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prm: TrainedProcessRewardModel,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 top_k=50,
                 top_p=0.95,
                 reward_alpha=1.0,
                 reward_confidence_default=1.0,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True,
                 force_overall_prompt_on_overall_question=True,
                 force_overall_question_on_overall_prompt=True,
                 random_reward=False,
                 logprob_reward=False,
                 ) -> None:
        super().__init__()
        self.base_model = base_model
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
        self.logprob_reward = logprob_reward
        self.prm = prm
        self.action_to_logprob = {}

        if random_reward:
            print("Random reward is enabled, the reward will be a random number between 0 and 1")

    def update_example(self, example: str, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt=prompt)

        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example + '\n\n')
            
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()
            self.overall_question = ""

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(self.prompt_examples)
            f.write(self.prompt["question_prefix"].format(question=self.example) + "\n")
            for idx, (step, _) in enumerate(state):
                is_last_step = idx == len(state) - 1
                f.write(
                    self.prompt["step_prefix"].format(idx=idx + 1) + " " + step)
                if not is_last_step:
                    f.write("\n")

            f.write(('\n' if not f.getvalue().endswith('\n') else '') + self.prompt["step_prefix"].format(idx=len(state) + 1))
            model_input = f.getvalue()

        outputs = []

        sampling_params = SamplingParams(temperature=self.temperature, 
                                          max_tokens=512, 
                                          n=self.n_actions, 
                                          stop=['.\n', '\n', '.\n\n', '\n\n'],
                                          include_stop_str_in_output=True)
        
        outputs = self.base_model.generate(model_input,
                                            sampling_params=sampling_params,
                                            use_tqdm=False,
                                            )[0].outputs
        
        outputs = [o.text for o in outputs]
        outputs = [self.process_action(o) for o in outputs]
        return outputs
    
    def get_actions_batch(self, states: list[GSM8kState]) -> list[list[GSM8kAction]]:
        """Batch version of get_actions that processes multiple states at once"""
        model_inputs = []
        
        # Build all model inputs first
        for state in states:
            with io.StringIO() as f:
                f.write(self.prompt_examples)
                f.write(self.prompt["question_prefix"].format(question=self.example) + "\n")
                for idx, (step, _) in enumerate(state):
                    is_last_step = idx == len(state) - 1
                    f.write(
                        self.prompt["step_prefix"].format(idx=idx + 1) + " " + step)
                    if not is_last_step:
                        f.write("\n")
                if self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                    f.write(' ' + self.prompt["answer_prefix"])
                else:
                    f.write(('\n' if not f.getvalue().endswith('\n') else '') + self.prompt["step_prefix"].format(idx=len(state) + 1))

                model_inputs.append(f.getvalue())

        sampling_params = SamplingParams(temperature=self.temperature,
                                      max_tokens=512,
                                      n=self.n_actions,
                                      stop=['.\n', '\n', '.\n\n', '\n\n'],
                                      include_stop_str_in_output=True,
                                      logprobs=1)

        # Single batch call to generate
        outputs = self.base_model.generate(model_inputs,
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
        
        assert len(states) == len(outputs)

        all_actions = []
        for state, output in zip(states, outputs):
            actions = [self.process_action(o.text) for o in output.outputs]
            action_logprobs = [o.logprobs for o in output.outputs] # list of list of dicts 
            for action, logprob_dict_list in zip(actions, action_logprobs):
                avg_logprob = 0
                for logprob_dict in logprob_dict_list:
                    for k, v in logprob_dict.items():
                        avg_logprob += v.logprob
                avg_logprob /= len(logprob_dict_list)
                state_repr = '-'.join([step for step, _ in state])
                self.action_to_logprob[(action, state_repr)] = avg_logprob
            
            all_actions.append(actions)

        assert len(all_actions) == len(states)
        return all_actions

    def process_action(self, action: str) -> str:
        """Process a single action output from the model"""
        # First get just the first line
        action = action.split('\n')[0].strip()
        
        # Remove any additional question if present
        if 'Question:' in action:
            action = action.split('Question:')[0].strip()
        
        # Handle answer sentences
        ans_sentence = re.findall(r'The answer is .*?[.]', action)
        if len(ans_sentence) > 0:
            ans_sentence = ans_sentence[0]
            action = action.split(ans_sentence)[0] + ans_sentence
            
        return action

    def fast_reward(self, state: GSM8kState, action: GSM8kAction, **kwargs) -> tuple[float, dict]:
        if self.random_reward:
            r = random.random()
            return r, {'r_useful': r}

        if self.logprob_reward:
            state_repr = ''.join([step for step, _ in state])
            r = self.action_to_logprob[(action, state_repr)]
            return r, {'r_useful': r}

        # Prepare the question and steps for the PRM
        question = self.example
        steps = [step for step, _ in state] + [action]
        
        # Use the PRM to predict correctness
        r_correct, info = self.prm.predict_correctness(question, steps)    
        # Return the predicted reward and a dictionary with the useful reward
        return r_correct, {'r_useful': r_correct}
        
    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful, 'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction, **kwargs) -> tuple[float, dict]:
        r_correct, _ = self.fast_reward(state, action, **kwargs)
        return r_correct, {'r_useful': r_correct}


    def fast_reward_batch(self, states: list[GSM8kState], actions: list[GSM8kAction], **kwargs) -> list[tuple[float, dict]]:
        if self.random_reward:
            rewards = [random.random() for _ in states]
            return [(r, {'r_useful': r}) for r in rewards]
        
        if self.logprob_reward:
            rewards = []
            for state, action in zip(states, actions):
                state_repr = '-'.join([step for step, _ in state])
                rewards.append(self.action_to_logprob[(action, state_repr)])
            return [(r, {'r_useful': r}) for r in rewards]

        # Prepare batched questions and steps for the PRM
        questions = [self.example] * len(states)
        steps_batch = [[step for step, _ in state] + [action] for state, action in zip(states, actions)]

        # Call predict_correctness_batch for all examples at once
        r_corrects = self.prm.predict_correctness_batch(questions, steps_batch)

        # Return list of rewards and info dicts
        return [(r, {'r_useful': r}) for r in r_corrects]

    def reward_batch(self, states: list[GSM8kState], actions: list[GSM8kAction], **kwargs) -> list[tuple[float, dict]]:
        rewards_and_infos = self.fast_reward_batch(states, actions, **kwargs)
        return [(r, info) for r, info in rewards_and_infos]