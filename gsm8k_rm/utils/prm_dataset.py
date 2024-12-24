import json, os, pickle as pkl
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from dataclasses import dataclass
import world_model
import random
import re
from collections import defaultdict
import datasets
from .gsm8k_utils import gsm8k_is_correct, retrieve_answer_from_dataset, retrieve_answer, judge_answer
from reasoners.benchmark.mathtask import is_correct as math_is_correct



def clean_traj(traj_str):
    ## remove all that comes after "The answer is"
    ### remove 'Step x:'
    traj_str = re.sub(r'\nStep \d+:', '', traj_str).strip()
    traj_str = re.sub(r'Step \d+:', '', traj_str).strip()
    traj_str = traj_str.replace('The answer is:', 'The answer is').strip()
    ## make sure there is a dot before "The answer is". If there is, do nothing
    traj_str = traj_str.replace('. The answer is', ' The answer is').replace(' The answer is', '. The answer is')
    ## remove all substrings that are <<xxx>> 
    traj_str = re.sub(r'<<.*?>>', '', traj_str).strip()
    traj_str = traj_str.split('\n\n')[0].strip()
    
    return traj_str

@dataclass
class Node():
    action: str
    reward: float
    state: str
    history: list
    cum_rewards: list
    return_to_go: float
    Q: float
    is_successful: bool


class PRMTrajectoryDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train'
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.split = split
        self.step_sep = ' ки'
        self.pos_step_token = '+'
        self.neg_step_token = '-'
        self.step_sep_id = self.tokenizer.encode(self.step_sep, add_special_tokens=False)[-1]
        self.pos_step_id = self.tokenizer.encode(self.pos_step_token, add_special_tokens=False)[-1]
        self.neg_step_id = self.tokenizer.encode(self.neg_step_token, add_special_tokens=False)[-1]
        self.num_samples = getattr(self.config, 'num_samples', 100000)
    
        self.data = self._load_data()
            
    def _load_data(self):
        all_trajs = []
        
        for path in self.data_path:
            for root, _, files in os.walk(path):
                for file in tqdm(files, desc='Loading data...'):
                    if file.endswith('.pkl'):
                        try:
                            with open(os.path.join(root, file), 'rb') as f:
                                task = 'gsm8k' if 'gsm8k' in root.lower() else 'MATH'
                                content = pkl.load(f)
                                ### add task information to all examples
                                for c in content:
                                    c['example']['task'] = task
                                all_trajs.extend(content)
                        
                        except Exception as e:
                            print(f"Error loading {os.path.join(root, file)}: {e}")
                            continue
                        
                        if self.config.debug:
                            all_trajs = all_trajs[:100]
                            break
        
        if not all_trajs:
            raise ValueError(f'No data found in {self.data_path}')

        if 'trajs' in all_trajs[0]:
            ## flatten out data to {question, answer, traj}
            data = []
            for traj in all_trajs:
                for t in traj['trajs']:
                    data.append({
                        'example': traj['example'],
                        'nodes': t,
                        'task': traj['example']['task']
                    })
        else:
            data = all_trajs

        ## filter out repeated trajectories
        print("Filtering out duplicate trajectories...")
        traj_str = set()
        new_data = []

        for traj in data:
            traj_id = (traj['example']['question'], "\n".join([node.action for node in traj['nodes']]))
            if traj_id not in traj_str:
                new_data.append(traj)
                traj_str.add(traj_id)

        print("Removed {} duplicate trajectories".format(len(data) - len(new_data)))
        data = new_data

        #### keep 100K traj at most from each task
        gsm8k_data = [d for d in data if d['example']['task'] == 'gsm8k']
        math_data = [d for d in data if d['example']['task'] == 'MATH']

        random.seed(42)
        if len(gsm8k_data) > self.num_samples:
            gsm8k_data = random.sample(gsm8k_data, self.num_samples)
        if len(math_data) > self.num_samples:
            math_data = random.sample(math_data, self.num_samples)

        data = gsm8k_data + math_data
        #### construct nodes from data--also to avoid reference issues
        new_data = []
        
        for traj in tqdm(data, desc='Processing data'):
            new_traj = []
            history = [] ## nodes before current node in the trajectory
            terminal_node = traj['nodes'][-1].state[-1].step

            ## throw away incomplete trajectories if filter_incomplete_trajs is true
            if getattr(self.config, 'filter_incomplete_trajs', True):
                if not any(phrase in terminal_node.lower() for phrase in ['the answer is', 'the final answer is']):
                    continue

            if traj['example']['task'] == 'gsm8k':
                is_successful = gsm8k_is_correct(terminal_node, traj['example']['answer'])
            elif traj['example']['task'] == 'MATH':
                is_successful = math_is_correct(terminal_node, traj['example']['answer'])
            else:
                raise ValueError(f"Unknown task: {traj['example']['task']}")

            for node in traj['nodes']:
                new_node = Node(
                    action=node.action,
                    reward=node.reward,
                    state=node.state,
                    history=[n for n in history],
                    cum_rewards=node.cum_rewards,
                    Q=node.Q,
                    return_to_go=None,
                    is_successful=is_successful
                )
                history.append(new_node)
                new_traj.append(new_node)
            
            new_data.append({
                'example': traj['example'],
                'nodes': new_traj
            })

        data = new_data
        return data
            
    def process_data(self, data):
        raise NotImplementedError

    def node2str(self, node):
        raise NotImplementedError

    def nodehistory2str(self, node):
        raise NotImplementedError      
    
    @staticmethod
    def tokenize_example(example, tokenizer, step_sep_id, pos_step_id, neg_step_id, max_length, config, split):
        question = example['question']
        steps_with_labels = example['steps_with_labels']
        # Tokenize question
        question_tokens = tokenizer.encode(question , add_special_tokens=False)
        
        input_ids = []
        labels = []
        loss_mask = []
        loss_mask_with_first_error_only = []
        # Add question tokens
        input_ids.extend(question_tokens)
        labels.extend([tokenizer.pad_token_id] * len(question_tokens))
        loss_mask.extend([0] * len(question_tokens))
        loss_mask_with_first_error_only.extend([0] * len(question_tokens)) ## Needed to compute TRACE error.
        after_first_error = False

        # Process each step
        for step_idx, step_info in enumerate(steps_with_labels):
            step = step_info['step']
            step_label = step_info['label']

            step = f'\nStep {step_idx+1}: {step}'
            
            # Tokenize step
            step_tokens = tokenizer.encode(step, add_special_tokens=False)
            input_ids.extend(step_tokens)
            labels.extend([tokenizer.pad_token_id] * len(step_tokens))
            loss_mask.extend([0] * len(step_tokens))
            loss_mask_with_first_error_only.extend([0] * len(step_tokens))
            
            # Add step separator
            input_ids.append(step_sep_id)
            labels.append(pos_step_id if step_label in ['+', 1] else neg_step_id)
            loss_mask.append(1)

            if not after_first_error:
                loss_mask_with_first_error_only.append(1)
            else:
                loss_mask_with_first_error_only.append(0)

            if labels[-1] == neg_step_id and not after_first_error:
                after_first_error = True

        assert len(input_ids) == len(labels) == len(loss_mask), "Input ids, labels, and loss mask should be the same length"
        assert len(input_ids) == len(loss_mask_with_first_error_only), "Input ids and loss mask with first error only should be the same length"
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        loss_mask = torch.tensor(loss_mask)
        loss_mask_with_first_error_only = torch.tensor(loss_mask_with_first_error_only)
        attention_mask = torch.ones_like(input_ids)

        # Truncate if necessary
        if len(input_ids) > max_length:
            #print("Warning: truncating input ids from {} to {}".format(len(input_ids), max_length))
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            loss_mask = loss_mask[:max_length]
            loss_mask_with_first_error_only = loss_mask_with_first_error_only[:max_length]
            attention_mask = attention_mask[:max_length]

        if getattr(config, 'distribute_final_answer_labels', False) and split == 'train':
            solution_label = example['solution_label']
            labels[loss_mask == 1] = pos_step_id if solution_label == 1 else neg_step_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask,
            'loss_mask_with_first_error_only': loss_mask_with_first_error_only
        }


    def _tokenize_example(self, example):
        return self.tokenize_example(example, self.tokenizer, self.step_sep_id, self.pos_step_id, self.neg_step_id, self.max_length, self.config, self.split)

    def __getitem__(self, idx):
        example = self.data[idx]
        return self._tokenize_example(example)

    def collate_fn(self, batch):
        input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        if 'loss_mask' in batch[0]:
            loss_mask = pad_sequence([b['loss_mask'] for b in batch], batch_first=True, padding_value=0)
            loss_mask_with_first_error_only = pad_sequence([b['loss_mask_with_first_error_only'] for b in batch], batch_first=True, padding_value=0)
            return_dict['loss_mask'] = loss_mask
            return_dict['loss_mask_with_first_error_only'] = loss_mask_with_first_error_only

        return return_dict
    
    def __len__(self):
        return len(self.data)
    
    def clean_trajectory(self, traj):
        return traj
        

class PRMPairwiseDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 ):
        
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        
        self.data = self.process_data(self.data)
            
    
    def process_data(self, data):
        """ Will return pairs of nodes that are at the same level and the label depends on whether the final answer is correct or not"""

        #### 1. group trajectories by question
        #### 2. for each question, find nodes across all trajectories that are at the same level i.e. same len(node.history) and have different is_successful values
        #### 3. label is 0 if first node is successful and second node is not, 1 if first node is not successful and second node is successful

        grouped_data = {}
        for d in data:
            question = d['example']['question']
            if question not in grouped_data:
                grouped_data[question] = []
            grouped_data[question].append(d)

        processed_data = []
        same_pairs = 0
        total_pairs = 0
        
        prefix_lengths = []
        for question, trajs in grouped_data.items():
            successful_nodes = []
            unsuccessful_nodes = []
            
            for i, traj in enumerate(trajs):
                for node in traj['nodes']:
                    if node.is_successful:
                        successful_nodes.append(node)
                    else:
                        unsuccessful_nodes.append(node)

            all_pairs = set() # to make sure we don't have duplicate pairs

            for s_node in successful_nodes:
                s_node_history_str = self.nodehistory2str(s_node)
                
                for u_node in unsuccessful_nodes:
                    u_node_history_str = self.nodehistory2str(u_node)
                    
                    if s_node_history_str == u_node_history_str:
                        if s_node.action == u_node.action:### if same action skip 
                            continue

                            
                        prefix_lengths.append(len(s_node.history))  

                        ### if ratio between action lengths is greater than 2.0 * succsseful action, skip 
                        if len(u_node.action) > 2.0 * len(s_node.action):
                            continue

                        node_pair = '-'.join((s_node.action, u_node.action))

                        if node_pair not in all_pairs:                
                            all_pairs.add(node_pair)
                            processed_data.append({
                                'traj_w': s_node,
                                'traj_l': u_node,
                                'question': question,
                                'answer': traj['example']['answer']
                            })
                        else:
                            same_pairs += 1
                    total_pairs += 1

        
        print("out of {} pairs, {} are same".format(total_pairs, same_pairs))
        ### histogram of prefix lengths: print percentage of prefixes of length 1, 2, 3, 4, 5, 6, 7
        print("Prefix lengths histogram:")
        prefix_lengths = np.array(prefix_lengths)
        for i in range(0, 7):
            print("Prefix length {}: {:.2f}%".format(i, 100 * np.mean(prefix_lengths == i)))
        
        return processed_data

    def node2str(self, node):
        prefix = "\n".join([n.state[-1].step for n in node.history]).strip()
        if prefix.strip():
            prefix += "\n"
              
        result = prefix + "Action: " + node.action
        return result 

    def nodehistory2str(self, node):
        result =  "\n".join([n.state[-1].step for n in node.history])
        return result
    
    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_traj = item['traj_w']
        rejected_traj = item['traj_l']
        question = item['question']

        #### tokenize each 
        chosen_traj_str = 'Q: ' + question + '\n' + self.node2str(chosen_traj)
        rejected_traj_str = 'Q: ' + question + '\n' + self.node2str(rejected_traj)

        ### log the trajectories
        if random.random() < 0.01:
            print(">>> Chosen traj:\n", chosen_traj_str)
            print(">>> Rejected traj:\n", rejected_traj_str)

        chosen_traj_tokenized = self.tokenizer(chosen_traj_str, padding=True, truncation=True, return_tensors='pt')
        rejected_traj_tokenized = self.tokenizer(rejected_traj_str, padding=True, truncation=True, return_tensors='pt')


        return {
            'input_ids_chosen': chosen_traj_tokenized['input_ids'][0],
            'attention_mask_chosen': chosen_traj_tokenized['attention_mask'][0],
            'input_ids_rejected': rejected_traj_tokenized['input_ids'][0],
            'attention_mask_rejected': rejected_traj_tokenized['attention_mask'][0],
        }

    def collate_fn(self, batch):
        input_ids_chosen = pad_sequence([b['input_ids_chosen'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_chosen = pad_sequence([b['attention_mask_chosen'] for b in batch], batch_first=True, padding_value=0)
        input_ids_rejected = pad_sequence([b['input_ids_rejected'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask_rejected = pad_sequence([b['attention_mask_rejected'] for b in batch], batch_first=True, padding_value=0)

        return {
            'input_ids_chosen': input_ids_chosen,
            'attention_mask_chosen': attention_mask_chosen,
            'input_ids_rejected': input_ids_rejected,
            'attention_mask_rejected': attention_mask_rejected,
        }
       
    def __len__(self):
        return len(self.data)
    

class PRMBinaryDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train'
                 ):
        
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length

        self.data = self.process_data(self.data)

    def process_data(self, data):
        """ Will return node and label where label is 1 if the final answer is correct and 0 otherwise"""

        grouped_data = {}
        prefix_lengths = [] 
        all_questions = set()

        for d in data:
            question = d['example']['question']
            all_questions.add(question)
            if question not in grouped_data:
                grouped_data[question] = []
            grouped_data[question].append(d)

        # Compute statistics
        traj_counts = [len(trajs) for trajs in grouped_data.values()]
        avg_traj_per_question = sum(traj_counts) / len(grouped_data)
        max_traj_per_question = max(traj_counts)
        min_traj_per_question = min(traj_counts)

        print(f"Statistics:")
        print(f"Average number of trajectories per question: {avg_traj_per_question:.2f}")
        print(f"Maximum number of trajectories per question: {max_traj_per_question}")
        print(f"Minimum number of trajectories per question: {min_traj_per_question}")
        print(f"Total number of unique questions: {len(grouped_data)}")
        print(f"Total number of trajectories: {sum(traj_counts)}")


        processed_data = []
        all_node_str = set()
        
        for question, trajs in grouped_data.items():
            for i, traj in enumerate(trajs):
                nodes = traj['nodes']
                try:
                    nstr = self.traj2str(nodes, task=traj['example']['task'])
                except Exception as e:
                    print("Warning: skipping trajectory for formatting issues.")
                    print(e)

                if nstr in all_node_str:
                    continue

                all_node_str.add(nstr)
                solution_label = int(nodes[-1].is_successful)
                steps_with_labels = []
                
                for step in nstr.split(self.step_sep):
                    step = step.strip()
                    if not step: continue
                    steps_with_labels.append({
                        'step': step,
                        'label': solution_label
                    })
                
                example = {
                    'traj': nstr,
                    'question': question,
                    'answer': traj['example']['answer'],
                    'solution_label': int(nodes[-1].is_successful),
                    'steps_with_labels': steps_with_labels
                }
                processed_data.append(example)
        
        print("Label distribution: ")
        print("Correct: {:.2f}%".format(100 * np.mean([d['solution_label'] for d in processed_data])))
        print("Incorrect: {:.2f}%".format(100 * np.mean([1 - d['solution_label'] for d in processed_data])))

        return processed_data

    def traj2str(self, nodes, task='gsm8k'):
        _nodes = [n for n in nodes if n.state[-1].step.strip()]
        traj_str = " ".join([n.state[-1].step + self.step_sep for n in _nodes]).strip()
        ## If "The answer is SOME NUMBER. xx" exists, remove XXXXX (sometimes model generates garbage)
        ### remove 'Step x:'
        traj_str = re.sub(r'Step \d+:', '', traj_str).strip()
        
        if task == 'gsm8k':
            traj_str = traj_str.replace('\n', '')
            ## remove any garbage after the answer.
            ans_sentence = re.findall(r'The answer is .*?[.]', traj_str)
            if len(ans_sentence) > 0:
                ans_sentence = ans_sentence[0]
                traj_str = traj_str.split(ans_sentence)[0] + ans_sentence + self.step_sep
            traj_str = traj_str.replace('. The answer is', ' The answer is').replace(' The answer is', '. The answer is')
        
        elif task == 'MATH':
            traj_str = traj_str.replace('.Final Answer', '. Final Answer')
        else:
            raise ValueError(f"Unknown task: {task}")
        
        ### make sure count of step_sep == number of nodes 
        assert traj_str.count(self.step_sep) == len(_nodes), "Number of step separators should be equal to number of nodes"
        return traj_str
        

class PRMContinuosScoreDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train', 
                 scale_rewards=True,
                 threshold_rewards=False
                 ):
        
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.scale_rewards = scale_rewards
        self.threshold_rewards = threshold_rewards
        
        self.data = self.process_data(self.data)

    def process_data(self, data):
        """ Will return node and label where label is 1 if the final answer is correct and 0 otherwise"""

        #### 1. group trajectories by question
        #### 2. for each question, find nodes across all trajectories that are at the same level i.e. same len(node.history) and have different is_successful values
        #### 3. label is 0 if first node is successful and second node is not, 1 if first node is not successful and second node is successful

        grouped_data = {}
        all_questions = set()

        for d in data:
            question = d['example']['question']
            all_questions.add(question)
            if question not in grouped_data:
                grouped_data[question] = []
            grouped_data[question].append(d)

        processed_data = []
        all_nodes = [] 

        #### find max and min reward to min-max scaling 
        all_rewards = []
        for question, trajs in grouped_data.items():
            for i, traj in enumerate(trajs):
                nodes = traj['nodes']
                rewards = [node.cum_rewards[0] for node in nodes]
                all_rewards.extend(rewards)

        self.min_reward = min(all_rewards)
        self.max_reward = max(all_rewards)
        self.min_reward = min(all_rewards)
        self.max_reward = max(all_rewards)
        ### compute mean and std of rewards
        self.mean_reward = np.mean(all_rewards)
        self.std_reward = np.std(all_rewards)
        self.mean_reward = np.mean(all_rewards)
        self.std_reward = np.std(all_rewards)

        ## print stats: 
        print("Reward stats:")
        print("Min reward: {:.2f}".format(self.min_reward))
        print("Max reward: {:.2f}".format(self.max_reward))
        print("Mean reward: {:.2f}".format(self.mean_reward))
        print("Std reward: {:.2f}".format(self.std_reward))
        print("Min reward: {:.2f}".format(self.min_reward))
        print("Max reward: {:.2f}".format(self.max_reward))
        print("Mean reward: {:.2f}".format(self.mean_reward))
        print("Std reward: {:.2f}".format(self.std_reward))

        all_node_str = set()
        for question, trajs in grouped_data.items():
            for i, traj in enumerate(trajs):
                nodes = traj['nodes']
                step_labels = [node.cum_rewards[0] for node in nodes]
                ## min-max scaling
                if self.scale_rewards:
                    step_labels = [(r - self.min_reward) / (self.max_reward - self.min_reward) for r in step_labels]
                if self.scale_rewards:
                    step_labels = [(r - self.min_reward) / (self.max_reward - self.min_reward) for r in step_labels]

                nstr = self.traj2str(nodes)
                if nstr in all_node_str:
                    continue

                all_node_str.add(nstr)
                all_nodes.append(nodes)
                
                example = {
                    'traj': nstr,
                    'question': question,
                    'answer': traj['example']['answer'],
                    'solution_label': int(nodes[-1].is_successful), # whether the answer was reached at the final node in the trajectory
                    'step_labels': step_labels
                }
                
                processed_data.append(example)
        
        print("Label distribution: ")
        print("Correct: {:.2f}%".format(100 * np.mean([d['solution_label'] for d in processed_data])))
        print("Incorrect: {:.2f}%".format(100 * np.mean([1 - d['solution_label'] for d in processed_data])))

        ### recompute reward stats after scaling
        if self.scale_rewards:
            all_rewards = []
            for d in processed_data:
                all_rewards.extend(d['step_labels'])

            self.min_reward = min(all_rewards)
            self.max_reward = max(all_rewards)
            ### compute mean and std of rewards
            self.mean_reward = np.mean(all_rewards)
            self.std_reward = np.std(all_rewards)

            ## print stats: 
            print("Reward stats after scaling:")
            print("Min reward: {:.2f}".format(self.min_reward))
            print("Max reward: {:.2f}".format(self.max_reward))
            print("Mean reward: {:.2f}".format(self.mean_reward))
            print("Std reward: {:.2f}".format(self.std_reward))

        #### if should threshold rewards 
        if self.threshold_rewards:
            print("Thresholding rewards...")
            ####### once a step is met that is below the threshold (0.7), it and all the steps afterwards are considered incorrect. The ones before are considered correct
            for d in processed_data:
                step_labels = d['step_labels']
                #threshold = 0.60
                threshold = min(step_labels)  # threshold is the minimum reward, we treat the least confident step as the first major mistake
                for i, sl in enumerate(step_labels):
                    if sl <= threshold:
                        step_labels[i:] = [0.0] * (len(step_labels) - i)
                        step_labels[:i] = [1.0] * i
                        break
                d['step_labels'] = step_labels
                
        return processed_data

    def traj2str(self, nodes, task='gsm8k'):
        traj_str = " ".join([n.state[-1].step + self.step_sep for n in nodes]).strip()
        ## If "The answer is SOME NUMBER. xx" exists, remove XXXXX (sometimes model generates garbage)
        ### remove 'Step x:'
        traj_str = re.sub(r'Step \d+:', '', traj_str).strip()
        
        if task == 'gsm8k':
            traj_str = traj_str.replace('\n', '')
            ans_sentence = re.findall(r'The answer is .*?[.]', traj_str)
            if len(ans_sentence) > 0:
                ans_sentence = ans_sentence[0]
                traj_str = traj_str.split(ans_sentence)[0] + ans_sentence + self.step_sep
            traj_str = traj_str.replace('. The answer is', ' The answer is').replace(' The answer is', '. The answer is')
        
        elif task == 'MATH':
            traj_str = traj_str.replace('. Final Answer', '. Final Answer')
        
        ### make sure count of step_sep == number of nodes 
        assert traj_str.count(self.step_sep) == len(_nodes), "Number of step separators should be equal to number of nodes"
        return traj_str
        
    def __getitem__(self, idx):
        item = self.data[idx]
        traj = item['traj']
        question = item['question']

        #### tokenize each 
        traj_str = 'Q: ' + question + '\nA: ' + traj
        ### log the trajectories
        
        if random.random() < 0.001:
            #print(">>> traj:\n", traj_str)
            step_labels = item['step_labels']
            _traj_str = traj_str
            for i, sl in enumerate(step_labels):
                lbl = "{:.2f}".format(sl)
                _traj_str = _traj_str.replace(self.step_sep, f" {lbl}", 1)

            print(">>> traj with labels:\n", _traj_str)

        traj_tokenized = self.tokenizer(traj_str, padding=True, truncation=True, return_tensors='pt', max_length=self.max_length)
        input_ids = traj_tokenized['input_ids'][0]
        attention_mask = traj_tokenized['attention_mask'][0]

        # loss mask is where self.step_sep is
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[input_ids == self.step_sep_id] = 1

        labels = torch.zeros_like(input_ids).float()
        if len(item['step_labels']) != len(input_ids[input_ids == self.step_sep_id]):
            print("Length mismatch between step labels and input ids")
            shorter_length = len(input_ids[input_ids == self.step_sep_id])
            item['step_labels'] = item['step_labels'][:shorter_length]

        labels[input_ids == self.step_sep_id] = torch.tensor(item['step_labels']).float()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask
        }

    def collate_fn(self, batch):
        input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        loss_mask = pad_sequence([b['loss_mask'] for b in batch], batch_first=True, padding_value=0)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask
        }
       
    def __len__(self):
        return len(self.data)
    
    def clean_trajectory(self, traj):
        return traj
    

class MathShepherdDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        
        super().__init__(data_path='', tokenizer=tokenizer, config=config, split=split)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.step_sep = ' ки'

        if getattr(self.config, 'distribute_final_answer_labels', False) and self.split == 'train':
            print("🤯 Will distribute final answer labels to intermediate steps for training 🤯")

        self.data = self.process_data(self.data)

    def _load_data(self):
        data = datasets.load_dataset('peiyi9979/Math-Shepherd', split='train')
        ## keep examples where gsm8k is the task 
        new_data = []
        task = self.config.task
        if isinstance(task, str):
            task = [task]
        
        for d in data:
            if d['task'].lower() in task:
                new_data.append(d)
            if self.config.debug and len(new_data) >= 500:
                break

        ### group examples by questions 
        grouped_data = defaultdict(list)
        for d in new_data:
            question = self.extract_question(d['input'], task=d['task'].lower())
            grouped_data[question].append(d)

        #### load the original GSM8K dataset and match the question to find the gold answer
        gold_data_gsm8k = datasets.load_dataset('gsm8k', 'main', split='train').to_list()
        with open('data/MATH/train.jsonl', 'r') as f:
            gold_data_math = [json.loads(line) for line in f]

        ## convert math dataset to dict problem --> question, solution --> answer
        gold_data_math = [{'question': d['problem'], 'answer': d['solution']} for d in gold_data_math]
        ## combine the two datasets
        gold_data = gold_data_math + gold_data_gsm8k
        gold_data = {d['question']: d['answer'] for d in gold_data}

        #import ipdb; ipdb.set_trace()
        questions_to_remove = []
        for q in grouped_data:
            _q = q
            if _q in gold_data:
                for d in grouped_data[q]:
                    d['answer'] = gold_data[_q]
            else:
                #print("Question {} not found in gold dataset, will remove...".format(q))
                questions_to_remove.append(q)

        print("removing samples from {} questions".format(len(questions_to_remove)))

        for q in questions_to_remove:
            del grouped_data[q]

        ### if train return data from first 90% of the questions
        # if test return data from last 10% of the questions 
        all_questions = list(grouped_data.keys())
        train_questions = all_questions[:int(0.9* len(all_questions))]
        test_questions = all_questions[int(0.9* len(all_questions)):]

        if self.split == 'train':
            new_data = []
            for q in train_questions:
                new_data.extend(grouped_data[q])
        else:
            new_data = []
            ### take at most 3 examples from each question
            for q in test_questions:
                new_data.extend(grouped_data[q][:3])
        
        return new_data
    
    def extract_question(self, traj, task):
        if task == 'math':
            return traj.split('Step 1:')[0].strip()
        elif task == 'gsm8k':
            if '?' in traj:
                return traj.split('?')[0].strip() + '?'
            elif 'Step 1:' in traj:
                return traj.split('Step 1:')[0].strip()
            else:
                return 'SOMEQUESTION'
    
    def process_data(self, data):
        #gold_data = datasets.load('gsm8k', split='train')
        processed_data = []
        for d in data: 
            task = d['task'].lower()
            if 'Step 1:' in d['input']:
                ## question is everything before Step 1: and traj is everything after
                question = d['input'].split('Step 1:')[0].strip()
                traj = 'Step 1:' + d['input'].split('Step 1:')[1].strip()
                label = 'Step 1:' + d['label'].split('Step 1:')[1].strip()
            else:
                continue

            label = label.replace('\n+', '\n +').replace('\n-', '\n -')
            
            traj = self.process_traj(traj, task=task)
            label = self.process_traj(label, task=task)

            steps_and_labels = self.extract_steps_and_labels(traj, label)


            if not steps_and_labels:
                #print("Skipping example because steps and labels could not be extracted")
                continue

            steps_and_labels[-1]['step'] = self.process_final_answer(steps_and_labels[-1]['step'], task=task)

            ## computing final answer correctness
            gold_chain = d['answer']
            try:
                is_traj_correct = steps_and_labels[-1]['label'] == '+'
            except:
                import ipdb; ipdb.set_trace()
            
            processed_data.append({
                'question': question,
                'traj': traj,
                'steps_with_labels': steps_and_labels,
                'solution_label': 1 if is_traj_correct else 0,
                'answer': gold_chain,
                'task': task
            })

        return processed_data

    def process_traj(self, traj_str, task='gsm8k'):
        ## remove all that comes after "The answer is"
        ### remove 'Step x:'
        traj_str = re.sub(r'\nStep \d+:', '', traj_str).strip()
        traj_str = re.sub(r'Step \d+:', '', traj_str).strip()
      
        ## remove all substrings that are <<xxx>> 
        traj_str = traj_str.replace('The answer is:', 'The answer is').strip()
        traj_str = traj_str.replace('. The answer is', ' The answer is').replace(' The answer is', '. The answer is')
        
        if task == 'gsm8k':
            traj_str = re.sub(r'<<.*?>>', '', traj_str).strip()
            traj_str = traj_str.split('\n\n')[0].strip()
            
        return traj_str
    
    def process_final_answer(self, final_answer, task='gsm8k'):
        if task == 'math':
            #### check dollar signs
            final_answer = final_answer.replace('The final answer is', 'The answer is')
            pattern = r"The answer is (\w+)(?!\$)"
            # Check if the match doesn't already have $$ around it
            def replacer(match):
                # Get the XXXX part of the match
                answer = match.group(1)
                return f"The answer is ${answer}$"
            # Use sub to replace the pattern with $$ if not already present
            final_answer = re.sub(pattern, replacer, final_answer)
        
        return final_answer

    def extract_steps_and_labels(self, trajectory, label):
        traj_steps = trajectory.split('ки')
        result = []

        for step in traj_steps:
            step = step.strip()

            if not step:
                continue
            # Find the step in the label string
            step_start = label.find(step)
            if step_start == -1:
                continue
            # Find the label after the step
            label_start = step_start + len(step) + 1
            if label_start < len(label):
                step_label = label[label_start].strip()
                if step_label not in ['+', '-']:
                    label_start += 1
                    step_label = label[label_start].strip()
                    if step_label not in ['+', '-']:
                        return None
                
                if not step.endswith('.'):
                    step += '.'
                
                result.append({
                    'step': step,
                    'label': step_label
                })
            label = label[label_start + 1:]

        return result
        
    def clean_trajectory(self, traj):
        return traj

class FutureSamplingDataset(MathShepherdDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(tokenizer=tokenizer, config=config, split=split)

        if getattr(self.config, 'distribute_final_answer_labels', False) and self.split == 'train':
            print("🤯 Will distribute final answer labels to intermediate steps for training 🤯")

        # Ensure only one label type is used
        label_types = [
            self.config.use_math_shepherd_label,
            self.config.use_future_sampling_label,
            self.config.use_self_consistency_label
        ]
        if sum(label_types) != 1:
            raise ValueError("Exactly one step label type must be selected")
        
        label_type = 'math_shepherd' if self.config.use_math_shepherd_label else 'future_sampling' if self.config.use_future_sampling_label else 'self_consistency'
        print(f"💪💪💪 Using {label_type.upper()} step labels for future sampling 💪💪💪")

    def _load_data(self):
        ## load all .json files in the data_path
        if self.split == 'train':
            data = []
            for file in os.listdir(self.config.data_dir):
                if file.endswith('.json') and 'args' not in file:
                    with open(os.path.join(self.config.data_dir, file), 'r') as f:
                        loaded_data = json.load(f)
                        data.extend(loaded_data)
        
        else: ### eval/test
            data = []
            for file in os.listdir(self.config.data_dir):
                if file.endswith('.json') and 'args' not in file:
                    with open(os.path.join(self.config.data_dir, file), 'r') as f:
                        loaded_data = json.load(f)
                        data.extend(loaded_data)
        
        return data

    def process_data(self, data):
        ## data is a dictionary where keys are question@@@prefix and values future sampling information. We want to process such that each prefix will be
        #1. ectract all unique trajectories 
        unique_question_trajectories = set()
        for k, v in data.items():
            unique_question_trajectories.add((v['problem'], v['traj']))

        gold_data_gsm8k = datasets.load_dataset('gsm8k', 'main', split='train').to_list()
        gold_data = {d['question'][:30]: d['answer'] for d in gold_data_gsm8k}

        ## 2. for each trajectory, split into prefixes, and obtain labels for each prefix 
        processed_data = []
        n_failed = 0
        n_skipped = 0
        for (question, traj) in tqdm(unique_question_trajectories, desc="Processing future sampling data"):
            steps = [s.strip() for s in traj.split('ки')]
            steps_and_labels = []
            
            for i, step in enumerate(steps):
                if not step.strip():
                    continue
                
                step = step.strip()
                prefix = ' '.join(steps[:i+1])
                qpre = "@@@".join([question, prefix])
                
                if qpre not in data:
                    n_failed += 1 # some will be skipped due to parsing issues, but very few < 30
                    break
                
                prefix_info = data[qpre]
                if not step.endswith('.'):
                    step += '.'

                if self.config.use_math_shepherd_label:
                    label = prefix_info['prefix_gt_label']
                
                elif self.config.use_future_sampling_label:
                    if 'The answer is' in step:
                        ## Corner case: final step in a solution. Just compare the answer in the step with the answer in the prefix info directly -- no need to look at future samples. 
                        label = judge_answer(output=retrieve_answer(step), answer=prefix_info['answer'])
                    else:
                        #threshold = 0.1 # percentage of the future samples
                        label = int(prefix_info['n_correct'] >= 1)
                
                elif self.config.use_self_consistency_label:
                    if 'leads_to_common_answer' not in prefix_info:
                        continue
                    label = int(prefix_info['leads_to_common_answer'])
                else:
                    raise ValueError("No label type specified")
            
                steps_and_labels.append({
                    'step': step,
                    'label': '+' if label in [1, True, '+'] else '-',
                    'original_label': prefix_info['prefix_gt_label'],
                    'info': prefix_info,
                })

            if len(steps_and_labels) == 0:
                n_skipped += 1
                continue
            
            answer = gold_data[question[:30]]

            processed_data.append({
            'question': question,
            'traj': traj,
            'steps_with_labels': steps_and_labels,
            'solution_label': None,
            'answer': answer
            })

        print(f"Failed to process {n_failed} prefixes out of {len(unique_question_trajectories)}")
        print(f"Skipped {n_skipped} trajectories due to uncertainty in SC label")
        return processed_data
    
class ManuallyAnnotatedDataset(MathShepherdDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)

        ## load tsv data_path
        self.data_path = data_path


    def _load_data(self):
        ## load all .json files in the data_path
        import csv
        from nltk.tokenize import sent_tokenize
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                question = row['question']
                traj = row['trajectory'] 
                traj = traj.split('####')[0]
                traj = clean_traj(traj)
                steps = [s.strip() for s in sent_tokenize(traj) if s.strip()]
                # Start Generation Here
                steps = [s if s.endswith('.') else s + '.' for s in steps]
                step_labels = [int(label) for label in row['step_labels'].split(',') if label.strip()]
                if len(steps) != len(step_labels):
                    print(f"Skipping example {traj} because number of steps and step labels do not match")
                    continue

                steps_and_labels = [{'step': step, 'label': label} for step, label in zip(steps, step_labels)]
                data.append({
                    'question': question,
                    'traj': traj,
                    'steps_with_labels': steps_and_labels,
                    'solution_label': None,
                })

        return data


    def process_data(self, data):
        return data

class PRMCoTDataset(PRMTrajectoryDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)

        self.data = self.process_data(self.data)
        
    def _load_data(self):
        ## load all .json files in the data_path
        if self.split == 'train':
            data = []
            for file in os.listdir(self.config.data_dir):
                if file.endswith('.json') and 'args' not in file:
                    with open(os.path.join(self.config.data_dir, file), 'r') as f:
                        loaded_data = json.load(f)
                        data.extend(loaded_data)
        
        else: ### eval/test
            data = []
            for file in os.listdir(self.config.data_dir):
                if file.endswith('.json') and 'args' not in file:
                    with open(os.path.join(self.config.data_dir, file), 'r') as f:
                        loaded_data = json.load(f)
                        data.extend(loaded_data)
    
        print(f"Loaded {len(data)} initial data points")
            
        return data

    def process_data(self, data):
        ## data is a dictionary where keys are question@@@prefix and values future sampling information. We want to process such that each prefix will be
        processed_data = []
        n_skipped = 0
        n_total = 0
        max_cots_per_solution = self.config.max_cots_per_solution

        for example in data:
            problem = example['problem']
            prefix = example['prefix']
            gt_labels = example['traj_gt_labels'] # list of '+' or '-'
            gt_labels = ['+' if lbl in ['+', 1] else '-' for lbl in gt_labels]
            ## get gt_labels until and including the first '-' if there is one. If there's no '-' get all of them
            if not gt_labels:
                continue

            steps = prefix.split('\n')
            is_correct = gt_labels[-1] == '+'
            n_trajs_so_far = 0

            for generation in example['generations']:
                # Extract the final decision from the generation
                n_total += 1
                decisions = [s for s in generation.split('\n') if s.strip() and s.startswith('Step')]
                if '-' in gt_labels:
                    labels_until_error = gt_labels[:gt_labels.index('-') + 1]
                else:
                    labels_until_error = gt_labels

                if len(decisions) != len(steps) or any('correct? yes' not in d.lower() and 'correct? no' not in d.lower() for d in decisions):
                    n_skipped += 1
                    continue

                decisions = ['+' if 'correct? yes' in d.lower() else '-' for d in decisions]
                # Check if the decision matches all ground truth labels until the first '-'
                if all([(decision == gt_label) for decision, gt_label in zip(decisions, labels_until_error)]):
                    n_trajs_so_far += 1
                    if n_trajs_so_far > max_cots_per_solution:
                        break
                    
                    ### process cot, but finding the Step x:, where x is len(labels_until_error) + 1 and replacing the rest with Step x+1: The step is incorrect since it follows an incorrect step. 
                    incorrect_step_index = len(labels_until_error) - 1
                    cot_steps = [line for line in generation.split('\n') if line.startswith('Step')]
                    ## remove steps after incorrect_step_index
                    cot_steps = cot_steps[:incorrect_step_index+1]
                    ## add the rest of the steps with the cot "The step is incorrect since it follows an incorrect step."
                    cot_steps.extend([f'Step {i+1}: Follows an incorrect step. Correct? No.' for i in range(incorrect_step_index+1, len(steps))])

                    assert len(cot_steps) == len(steps) # sanity check

                    if getattr(self.config, 'direct_prm', False):
                        cot_steps = [re.sub(r'(Step .*:).*?(Correct\? (Yes|No))', r'\1 \2', step) for step in cot_steps]

                    cot = '\n'.join(cot_steps)    

                    processed_data.append({
                        'problem': problem,
                        'solution': prefix,
                        'cot': cot,
                        'solution_steps': steps,
                        'cot_steps': cot_steps,
                        'labels': labels_until_error,
                        'is_correct': is_correct
                    })                
                else:
                    n_skipped += 1

        print(f"Skipped {n_skipped}/{n_total} examples due to extraction errors/incorrect decisions")

        if self.config.balance_data:
            # Count total number of Yes/No labels across all examples
            all_labels = []
            for example in processed_data:
                all_labels.extend([1 if l == '+' else 0 for l in example['labels']])
            
            # Get counts and determine minority class
            yes_count = sum(all_labels)
            no_count = len(all_labels) - yes_count
            
            print(f"Before balancing - Yes labels: {yes_count}, No labels: {no_count}")
            
            minority_class = '+' if yes_count < no_count else '-'
            majority_class = '-' if minority_class == '+' else '+'
            
            # Find examples where all labels are the minority class
            minority_examples = []
            for example in processed_data:
                if all(label == minority_class for label in example['labels']):
                    minority_examples.append(example)
            
            # Calculate how many times to duplicate minority examples
            label_diff = abs(yes_count - no_count)
            # Calculate average number of labels per minority example
            labels_per_example = sum(len(ex['labels']) for ex in minority_examples) / len(minority_examples) if minority_examples else 1
            n_duplicates = label_diff // (labels_per_example * len(minority_examples)) if minority_examples else 0
            
            # Duplicate minority examples
            for _ in range(int(n_duplicates)):
                processed_data.extend(minority_examples)
            
            # Recount labels after balancing
            all_labels = []
            for example in processed_data:
                all_labels.extend([1 if l == '+' else 0 for l in example['labels']])
            yes_count = sum(all_labels)
            no_count = len(all_labels) - yes_count
            
            print(f"After balancing - Yes labels: {yes_count}, No labels: {no_count}")

        return processed_data

    def format_cot_data(self, problem, solution, cot=None):
        instruction = ("Given a math question and partial solution steps, analyze each step in the solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct.")
        try:
            if cot:
                return self.tokenizer.apply_chat_template([
                    {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                    {'role': "assistant", "content": f"Analysis:\n{cot}"}
                ], tokenize=False)
            else:
                s = self.tokenizer.apply_chat_template([
                    {'role': "user", "content": f"{instruction}\n\nQuestion: {problem}\n\n{solution}"},
                    {'role': "assistant", "content": ""}
                ], tokenize=False, add_generation_prompt=False)
                return s.replace(self.tokenizer.eos_token, '')
        except Exception:
            print("Failed to apply chat template, using fallback format")
            if cot:
                return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\nAnalysis:\n{cot}{self.tokenizer.eos_token}"
            else:
                return f"{instruction}\n\nQuestion: {problem}\n\n{solution}\n\nAnalysis:"
    
    def _tokenize_example(self, example):
        ret_dict = {}
        if 'cot' in example:  # Training example
            # Combine instruction, problem, prefix, and COT
            input_text = self.format_cot_data(example['problem'], example['solution'], example['cot'])
            tokenized = self.tokenizer(input_text, padding=True, truncation=False, return_tensors='pt', add_special_tokens=False)
            
            if len(tokenized['input_ids'][0]) > self.max_length:
                print(f"Truncating input_ids because it's too long: {len(tokenized['input_ids'][0])}")
                input_ids = tokenized['input_ids'][0][:self.max_length]
                attention_mask = tokenized['attention_mask'][0][:self.max_length]
            else:
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]

            # Find the start of the COT part
            cot_start = input_text.index("Analysis:")

            cot_tokens = self.tokenizer(input_text[cot_start:], return_tensors='pt')['input_ids'][0]
            # Set loss mask to 1 for the COT tokens

            # Create labels
            labels = torch.full_like(input_ids, -100)  # -100 is the ignore index for CrossEntropyLoss
            labels[-len(cot_tokens):] = input_ids[-len(cot_tokens):]

            ret_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }
        else:  # Evaluation example
            solution = example['solution']
            input_text = self.format_cot_data(example['problem'], solution)

            tokenized = self.tokenizer(input_text, padding=True, return_tensors='pt', max_length=self.max_length, add_special_tokens=False)
            input_ids = tokenized['input_ids'][0]
            attention_mask = tokenized['attention_mask'][0]

            ret_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }

            if 'traj_gt_labels' in example: # eval
                step_labels = [1 if lbl in ['+', 1] else 0 for lbl in example['traj_gt_labels']]
                step_labels = torch.tensor(step_labels)
                ret_dict['step_labels'] = step_labels
        
        return ret_dict
        
    def collate_fn(self, batch):
        return_dict = {}
        keys_to_pad = ['input_ids', 'attention_mask']
        
        for key in keys_to_pad:
            return_dict[key] = pad_sequence([b[key] for b in batch], batch_first=True, padding_value=self.tokenizer.pad_token_id if key == 'input_ids' else 0)
        
        if 'labels' in batch[0]:
            return_dict['labels'] = pad_sequence([b['labels'] for b in batch], batch_first=True, padding_value=-100)
        elif 'step_labels' in batch[0]:
            return_dict['step_labels'] = [b['step_labels'] for b in batch]

        return return_dict
    

class PRMCoTInterleavedDataset(PRMCoTDataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 split='train',
                 ):
        super().__init__(data_path=data_path, tokenizer=tokenizer, config=config, split=split)

    def format_inst_question(self, problem):
        ##### will train the model to predict the cot and label after each step before the solution is complete
        ##### this is helpful for beam search to score partial solutions
        instruction = ("Given a math question and steps from a partial solution. You should analyze each step in the solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct.")
        return f"{instruction}\n\nQuestion: {problem}\n\n"
    
    def _tokenize_example(self, example):
        ## this function will tokenize the instruction+question, then interleave solution steps with CoT analysis
        ## and only apply loss on the CoT tokens
        problem = example['problem']
        solution_steps = example['solution_steps']
        cot_steps = example['cot_steps']

        ## remove step \d: from cot steps
        cot_steps = [re.sub(r'Step \d: ', '', step) for step in cot_steps]
        
        # Tokenize instruction and question first
        inst_question = self.format_inst_question(problem)
        inst_question_tokens = self.tokenizer(inst_question, add_special_tokens=False)
        input_ids = inst_question_tokens['input_ids']
        attention_mask = inst_question_tokens['attention_mask']
        
        # Initialize labels with -100 (no loss) for instruction+question
        labels = [-100] * len(input_ids)
        
        # Interleave solution steps with CoT analysis
        for step, step_cot in zip(solution_steps, cot_steps):
            # Add solution step
            step_tokens = self.tokenizer(f"{step}\n", add_special_tokens=False)
            input_ids.extend(step_tokens['input_ids'])
            attention_mask.extend(step_tokens['attention_mask'])
            labels.extend([-100] * len(step_tokens['input_ids']))
            
            # Add CoT analysis
            cot_tokens = self.tokenizer(f"Analysis: {step_cot}\n", add_special_tokens=False)
            input_ids.extend(cot_tokens['input_ids'])
            attention_mask.extend(cot_tokens['attention_mask'])
            # Apply loss on CoT tokens
            labels.extend(cot_tokens['input_ids'])
        
        assert len(input_ids) == len(labels)
        assert len(input_ids) == len(attention_mask)
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        labels = torch.tensor(labels)

        ### truncate if necessary
        if len(input_ids) > self.max_length:
            print(f"Truncating input_ids because it's too long: {len(input_ids)}")
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
class PRMCoTEvalDataset(PRMCoTDataset):
    def __init__(self, 
                 examples: list, 
                 tokenizer, 
                 config=None, 
                 split='eval',
                 process_data: bool = True,
                 ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.config = config
        self.split = split

        super().__init__(data_path=None, tokenizer=tokenizer, config=config, split=split)
        if process_data:
            self.data = self.process_data(self.examples)
        
        self._validate_data(self.data)

    def _validate_data(self, data):
        for example in data:
            if any(k not in example for k in ['problem', 'traj_gt_labels', 'traj_steps']):
                raise ValueError("Invalid example format")
            
    def _load_data(self):
        return self.examples

    def process_data(self, data):
        processed_data = [] 
        for item in data:
            question = item['question']
            steps = [step_info['step'] for step_info in item['steps_with_labels']]
            traj_gt_labels = [step_info['label'] for step_info in item['steps_with_labels']]

            solution = '\n'.join([f'Step {j+1}: {step}' for j, step in enumerate(steps)])

            cot_example = {
                'problem': question,
                'traj_gt_labels': traj_gt_labels,
                'traj_steps': steps,
                'solution': solution
            }
            processed_data.append(cot_example)

        return processed_data
    


    



