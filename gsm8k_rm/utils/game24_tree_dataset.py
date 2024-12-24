import json, os, pickle as pkl
from tqdm import tqdm
import torch
import numpy as np
from .tree_dataset import SearchTreeDataset
from dataclasses import dataclass
from typing import NamedTuple
import sys 
import sympy
sys.path.append('examples/tot_game24')
from world_model import ToTNode
from utils.prompt_selector import PromptSelector
from decision_transformer.game24 import Game24Engine
from utils.callbacks import Game24EvalCallBack

class Game24SearchTreeDataset(SearchTreeDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 bin_boundaries=None,
                 intermediate_bin_boundaries=None,
                 evaluator=None,
                 eval=False):
        
      
        self.eval = eval
        self.task_prompt = PromptSelector.select_prompt(
            task = 'game24',
            model_name = tokenizer.name_or_path
        )
        
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
        
                        
        print('Loaded {} trajectories'.format(len(all_trajs)))

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
    

    def construct_prompts(self, data):
        """
        
        1. Will combine all trajectories for the same problem.
        2. Then depending on the trajectoriy sampling strategy will sample tree search trajectories and pass them to the promptize_trajs function to create a single prompt. 

        Trajectorie smapling strategies:
        * fixed-random: randomly sample K trajectories for each problem. 
        * fixed-incremental: sample K trajectories with increasing sum of rewards. 
        * variable-random: sample variable number of trajectories for each problem.
        * variable-incremental: sample variable number of trajectories with increasing sum of rewards. 


        In all strategies, the final trajecoctory should always be a successful unless no successful trajectory is available. 
        
        """

        problem_to_trajs = {}
        for traj in data:
            if traj['question'] not in problem_to_trajs:
                problem_to_trajs[traj['question']] = []
            problem_to_trajs[traj['question']].append(traj)

        all_prompts = []
        K = self.config.n_trajs_per_prompt
        n_samples_per_problem = self.config.get('n_samples_per_problem', 10)
        assert K >= 2, "n_trajs_per_prompt should be at least 2"

        for problem, trajs in tqdm(problem_to_trajs.items(), desc='Constructing prompts'):
            unsuccesful_trajs = [traj for traj in trajs if not traj['success'] and self.is_valid(traj)]
            succesful_trajs = [traj for traj in trajs if traj['success'] and self.is_valid(traj)]

            if len(unsuccesful_trajs) == 0 or len(succesful_trajs) == 0:
                continue

            ### if no successful trajecotry, use the highest reward trajectory as the successful one.
            #if len(succesful_trajs) == 0:
                ## skip 
            #    continue
                #succesful_trajs = sorted(unsuccesful_trajs, key=lambda x: x['total_return'])[-1:]
                ## remove it from unsuccesful trajs

            # K-1 should be less than or equal to the number of unsuccesful trajs
            K = min(K, len(unsuccesful_trajs) + 1)

            for _ in range(n_samples_per_problem):
                sampled_unsuccessful_trajs = np.random.choice(unsuccesful_trajs, K-1, replace=False).tolist()
                sampled_succesful_traj = np.random.choice(succesful_trajs, 1).tolist()
                
                if self.config.traj_sampling_strategy == 'fixed-random':
                    ### shuffle unsuccesful trajs 
                    np.random.shuffle(sampled_unsuccessful_trajs)
                    sampled_trajs = sampled_unsuccessful_trajs + sampled_succesful_traj

                elif self.config.traj_sampling_strategy == 'fixed-incremental':
                    sampled_unsuccessful_trajs = sorted(sampled_unsuccessful_trajs, key=lambda x: x['total_return']) 
                    sampled_trajs = sampled_unsuccessful_trajs + sampled_succesful_traj

                elif self.config.traj_sampling_strategy == 'variable-random':
                    KK = np.random.randint(1, len(sampled_unsuccessful_trajs))
                    np.random.shuffle(sampled_unsuccessful_trajs)
                    sampled_trajs = sampled_unsuccessful_trajs[:KK] + sampled_succesful_traj

                elif self.config.traj_sampling_strategy == 'variable-incremental':
                    KK = np.random.randint(1, len(sampled_unsuccessful_trajs))
                    sampled_trajs = sorted(sampled_unsuccessful_trajs, key=lambda x: x['total_return'])[:KK] + sampled_succesful_traj
                
                else:
                    raise ValueError("Invalid trajectory sampling strategy")

                prompt = self.promptize_trajs(sampled_trajs)
                all_prompts.append(prompt)

        #### filter out duplicate prompts 
        all_prompts = list({prompt['prompt_str']: prompt for prompt in all_prompts}.values())
        return all_prompts
    
    
    def promptize_trajs(self, trajs):
        '''
        will take in as input a list of tree search trajectories for a given problem. 
        Then, will promptize the search process over these K trajectories as follows: 

        I will now try the following solution [TRAJ 1]
        A + B = C, 
        will this step lead to 24? sure/likely/impossible
        D + E = F
        will this step lead to 24? sure/likely/impossible
        X + G = H
        will this step lead to 24? sure/likely/impossible

        That did not work. Let's try something else: [TRAJ 2]
        XXXX
        XXX

        That did not work. Let's try something else: [TRAJ 3]
        XXX
        XXX
        XXX

        That worked. 
        The correct solution is: 
        [Succsseful Trajectory]

        '''

        #assert trajs[-1]['success'], "Last trajectory should be successful"
        problem = trajs[0]['question']
        prompt = self.task_prompt.format(problem=problem)
        user_prompt = self.task_prompt.format(problem=problem)


        bin_to_action_reward_string = {
            1: 'impossible',
            2: 'likely',
            3: 'sure',
        }

        succesful_traj_exists = False
    
        for traj in trajs:
            prompt += f"I will now try the following solution: \n"

            for i, node in enumerate(traj['traj']):
                ### clean node action
                node.action = self.preprocess_action(node.action)

                prompt += f'{node.action} ' 
                prompt += f"\nlikelihood of obtaining 24: {bin_to_action_reward_string[node.reward]}\n"

                #if bin_to_action_reward_string[node.reward] == 'impossible':
                #    break

            prompt += '\n'
            if not traj['success']:
                prompt += 'That did not work.\n\n'
            else:
                prompt += 'That worked. \n\nThe correct solution is: \n'
                for node in traj['traj']:
                    prompt += f'{node.action} \n'

                succesful_traj_exists = True
                break
        
        if not succesful_traj_exists:
            prompt += 'No solution found.\n'

        
        prompt += "Done"

        if np.random.random() < 0.001:
            print(prompt)

        prompt_ids = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True)
        ### define loss mask to mask out the user prompt.
        loss_mask = np.array([1] * len(prompt_ids[0]))
        user_prompt_len = len(self.tokenizer.encode(user_prompt, return_tensors='pt', truncation=True)[0])
        loss_mask[:user_prompt_len] = [0] * user_prompt_len
        loss_mask = loss_mask.tolist()

        ret_dict = {
            'prompt_str': prompt,
            'loss_mask': loss_mask,
            'token_ids': prompt_ids[0],
            'question': problem,
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
        if 'traj_str' not in traj:
            traj['traj_str'] = '\n'.join([node.action for node in traj['traj']])
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
