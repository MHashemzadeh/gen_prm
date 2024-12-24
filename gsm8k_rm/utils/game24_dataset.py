import json, os, pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
from .traj_dataset import ReasoningTrajectoryDataset
from dataclasses import dataclass
from typing import NamedTuple
import sys 
import sympy
sys.path.append('examples/tot_game24')
from world_model import ToTNode
from utils.answer_utils import GAME24_PROMPT, BIN_TO_RETURN_TOKEN, ACTION_SEP, VALUE_TO_RETURN_STR, BIN_TO_INTERMEDIATE_TOKEN, GAME24_INSTRUCTION_PROMPT, GAME24_INSTRUCTION_SCORE_PROMPT
from decision_transformer.game24 import Game24Engine
from utils.callbacks import Game24EvalCallBack

class Game24TrajectoryDataset(ReasoningTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 bin_boundaries=None,
                 intermediate_bin_boundaries=None,
                 evaluator=None,
                 instruction_tuning_mode=False,
                 eval=False):
        
        if instruction_tuning_mode:
            if config.quantize_reward:
                self.tokenize_traj = self.tokenize_traj_instruction_quantize
            else:
                self.tokenize_traj = self.tokenize_traj_instruction_score
        else:
            self.tokenize_traj = self.tokenize_traj_dt

        self.eval = eval
        
        super().__init__(data_path, tokenizer, config, 
                            bin_boundaries=bin_boundaries, 
                            intermediate_bin_boundaries=intermediate_bin_boundaries,
                            evaluator=evaluator,
                            end_of_traj_str='done')

    def _load_data(self):
        all_trajs = []
        for root, _, files in os.walk(self.data_path):
            for file in tqdm(files, desc='Loading data'):
                if file.endswith('.pkl'):
                    with open(os.path.join(root, file), 'rb') as f:
                        try:
                            _ = pkl.load(f)
                        except Exception as e:
                            print(e)
                            print(">>> Error loading {}".format(os.path.join(root, file)))
                            continue
                        all_trajs.extend(_)
                        if self.config.get('debug', False):
                            break
        
        #### divide rewards by the maximum reward
        #max_reward = max([max([node.reward for node in traj['traj']]) for traj in all_trajs])
        ### filter duplicate trajectories
                        
        print('Loaded {} trajectories'.format(len(all_trajs)))



        #### keep only the first three actions in each trajectory (to avoid invalid trajectories) 
        print("Keeping only the first three actions in each trajectory")
        filtered_trajs = []
        for traj in all_trajs:
            traj['traj'] = traj['traj'][:3]
            traj['rewards'] = traj['rewards'][:3]
            traj['actions'] = traj['actions'][:3]
            filtered_trajs.append(traj)

        all_trajs = filtered_trajs

        print("Filtering duplicate trajectories...")
        trajs_set = set()
        filtered_trajs = []
        for traj in all_trajs:
            traj_str = '{} --- {}'.format(traj['question'], '\n'.join([node.action for node in traj['traj']]))   
            if traj_str not in trajs_set:
                trajs_set.add(traj_str)
                filtered_trajs.append(traj)
        
        print('Filtered {} duplicate trajectories'.format(len(all_trajs) - len(filtered_trajs)))
        all_trajs = filtered_trajs

        ### recompute success because the ones in the data are not fully reliable. 
        for traj in all_trajs:
            traj['success'] = Game24EvalCallBack.eval_dt_output_with_left_numbers(problem=traj['question'], output='\n'.join([node.action for node in traj['traj']]))

        ### if self.eval, keep only question that are not solved by the teacher model
        if self.eval:
            print("Filtering solved questions for eval set...")
            solved_questions = set()
            for traj in all_trajs:
                if traj['success']:
                    solved_questions.add(traj['question'])
            all_trajs = [traj for traj in all_trajs if not traj['question'] in solved_questions]


        ### assign every action in succssful trajectories the maximum reward
        max_action_reward = max([max([node.reward for node in traj['traj']]) for traj in all_trajs])
        for traj in all_trajs:
            if traj['success']:
                for node in traj['traj']:
                    node.reward = max_action_reward
                traj['rewards'] = [max_action_reward] * len(traj['rewards'])

                
        return all_trajs
    
    def tokenize_traj_dt(self, traj):
        '''
        returns tokenization
          QUESTION: <question> [SEP] REWARD STATE ACTION REWARD STATE ACTION....
        '''
        problem = traj['question']
        traj_str = GAME24_PROMPT.format(problem=problem)
        traj_token_ids = self.tokenizer.encode(traj_str, add_special_tokens=False)
        traj_loss_mask = [0] * len(traj_token_ids)
        returns = []

        for i, node in enumerate(traj['traj']):
            return_to_go = round(node.return_to_go, self.reward_precision)
            returns.append(return_to_go)
            
            if self.config.quantize_reward:
                bin = self.get_bin(return_to_go)
                ## get special token for this bin
                if i == 0: 
                    ### if successful, assign to the top bin
                    if traj['success']:
                        bin = self.config.quantize_reward_bins
                    else:
                        token_str = BIN_TO_RETURN_TOKEN[bin]

                elif self.config.get('use_intermediate_tokens'):
                    token_str = BIN_TO_INTERMEDIATE_TOKEN[bin]
                else:
                    token_str = ''
                #token_str = BIN_TO_RETURN_TOKEN[bin] if (self.config.get('use_intermediate_tokens', True) or i == 0) else ''
                s = token_str.format(bin) + ACTION_SEP
            else:
                s = VALUE_TO_RETURN_STR(return_to_go) + ACTION_SEP

            if s.strip() and 'Answer:' not in node.action: ## don't add reward token if 'Answer:' is in the action
                tokens = self.tokenizer.encode(
                    s,
                    add_special_tokens=False
                )
                traj_str += s
                traj_token_ids.extend(tokens)
                mask = [0 if i == 0 else 1] * len(tokens)

                traj_loss_mask.extend(mask)

            ### clean node action
            node.action = self.preprocess_action(node.action)
                
            s = f'{node.action}' + ACTION_SEP
            if i == len(traj['traj']) - 1:
                s += self.end_of_traj_str.strip()
            
            tokens = self.tokenizer.encode(
                s,
                add_special_tokens=False)
            
            traj_str += s
            traj_token_ids.extend(tokens)
            mask = [1] * len(tokens)

            traj_loss_mask.extend(mask)

        assert len(traj_token_ids) == len(traj_loss_mask)

        if np.random.random() < 0.01:
            print(traj_str)

        if len(traj_token_ids) > self.max_length:
            traj_token_ids = traj_token_ids[:self.max_length]
            traj_loss_mask = traj_loss_mask[:self.max_length]
        
        ret_dict = {
            'traj_str': traj_str,
            'token_ids': traj_token_ids,
            'loss_mask': traj_loss_mask,
            'returns': returns,
            'question': problem,
            'is_correct': traj['success'],
        }

        return ret_dict
    

    def tokenize_traj_instruction_quantize(self, traj):
        '''
        returns tokenization
          QUESTION: <question> [SEP] REWARD STATE ACTION REWARD STATE ACTION....
        '''
        problem = traj['question']

        if not self.config.correct_only:
            assert self.config.quantize_reward_bins == 2, "only support 2 bins (Close/Not Close) for instruction tuning prompts"

        solution_type = None 
        if traj['success']:
            solution_type = 'Correct'
        else:
            ## Close if bin 2 else Not Close
            solution_type = 'Close' if self.get_bin(traj['traj'][0].return_to_go) == 2 else 'Not Close'

        traj_str = GAME24_INSTRUCTION_PROMPT.format(problem=problem, type=solution_type)
        
        traj_token_ids = self.tokenizer.encode(traj_str, add_special_tokens=False)
        traj_loss_mask = [0] * len(traj_token_ids)
        returns = []

        for i, node in enumerate(traj['traj']):
            return_to_go = round(node.return_to_go, self.reward_precision)
            returns.append(return_to_go)
            
            ### clean node action
            node.action = self.preprocess_action(node.action)
                
            s = f'{node.action}' + '\n'
            if i == len(traj['traj']) - 1:
                s += ' ' + self.end_of_traj_str.strip()
            
            tokens = self.tokenizer.encode(
                s,
                add_special_tokens=False)
            
            traj_str += s
            traj_token_ids.extend(tokens)
            mask = [1] * len(tokens)

            traj_loss_mask.extend(mask)

        assert len(traj_token_ids) == len(traj_loss_mask)

        if np.random.random() < 0.001:
            print(traj_str)

        if len(traj_token_ids) > self.max_length:
            traj_token_ids = traj_token_ids[:self.max_length]
            traj_loss_mask = traj_loss_mask[:self.max_length]
        
        ret_dict = {
            'traj_str': traj_str,
            'token_ids': traj_token_ids,
            'loss_mask': traj_loss_mask,
            'returns': returns,
            'question': problem,
            'is_correct': traj['success'],
        }

        return ret_dict
    
    def tokenize_traj_instruction_score(self, traj):
        '''
        returns tokenization
          QUESTION: <question> [SEP] REWARD STATE ACTION REWARD STATE ACTION....
        '''
        problem = traj['question']

        score = round(traj['traj'][0].return_to_go, self.reward_precision)
    
        traj_str = GAME24_INSTRUCTION_SCORE_PROMPT.format(problem=problem, score=score,
                                                          max_score=round(self.max_return_to_go, self.reward_precision))
        
        traj_token_ids = self.tokenizer.encode(traj_str, add_special_tokens=False)
        traj_loss_mask = [0] * len(traj_token_ids)
        returns = []

        for i, node in enumerate(traj['traj']):
            return_to_go = round(node.return_to_go, self.reward_precision)
            returns.append(return_to_go)
            
            ### clean node action
            node.action = self.preprocess_action(node.action)
                
            s = f'{node.action}' + '\n'
            if i == len(traj['traj']) - 1:
                s += ' ' + self.end_of_traj_str.strip()
            
            tokens = self.tokenizer.encode(
                s,
                add_special_tokens=False)
            
            traj_str += s
            traj_token_ids.extend(tokens)
            mask = [1] * len(tokens)

            traj_loss_mask.extend(mask)

        assert len(traj_token_ids) == len(traj_loss_mask)

        if np.random.random() < 0.01:
            print(traj_str)

        if len(traj_token_ids) > self.max_length:
            traj_token_ids = traj_token_ids[:self.max_length]
            traj_loss_mask = traj_loss_mask[:self.max_length]
        
        ret_dict = {
            'traj_str': traj_str,
            'token_ids': traj_token_ids,
            'loss_mask': traj_loss_mask,
            'returns': returns,
            'question': problem,
            'is_correct': traj['success'],
        }

        return ret_dict


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['token_ids'])
        attention_mask = torch.tensor([1] * len(item['token_ids']))
        loss_mask = torch.tensor(item['loss_mask'])
        labels = input_ids.clone()
        
        labels[loss_mask == 0] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def is_correct(self, traj):
        return traj['success']
    

    def is_valid(self, traj):
        ## to double check if the trajecotory is valid i.e., left numbers are correct. 
        output = traj['traj_str']
        output = output.split('done')[0].split('Answer:')[0]
        formulas = [line for line in output.split('\n') if '=' in line]
        cur_left_numbers = traj['question']

        for f in formulas:
            ## make sure left numbers are good
            try:
                left_numbers = Game24Engine.get_left_numbers(f, cur_left_numbers)
            except Exception as e:
                #("error with expression: {} given left numbers: {}".format(f, cur_left_numbers))
                return False
            cur_left_numbers = left_numbers
        return True
    

    def preprocess_action(self, action):
        ### remove ~ that GPT-4 outputs sometimes. 
        action = action.replace('~', '')
        return action
