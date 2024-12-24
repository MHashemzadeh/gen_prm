import json, os, pickle as pkl
from torch.utils.data import Dataset, DataLoader
from reasoners.algorithm import MCTS, MCTSNode
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from .answer_utils import get_action_trace_from_plan_str
import copy
from typing import NamedTuple
from dataclasses import dataclass
from utils.answer_utils import GAME24_PROMPT, BIN_TO_RETURN_TOKEN


@dataclass
class Node():
    action: str
    reward: float
    state: str
    cum_rewards: list
    return_to_go: float
    Q: float


def build_bin_boundaries(values, method='histogram', n_bins=10):
    if method == 'histogram':
        bin_boundaries = torch.linspace(min(values), max(values), n_bins + 1)
    elif method == 'percentile':
        bin_boundaries = torch.tensor(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
    else:
        raise ValueError('quantize_reward_method must be either histogram or percentile')
    
    assert len(bin_boundaries) == n_bins + 1
    return bin_boundaries


def normalize_returns(data, length_penalty=1.0):
    #### normalize by length then standarsize
    all_returns = [] 
    for traj in data:
        for i, node in enumerate(traj['traj']):
            ## normalize by number of remaining steps 
            node.return_to_go /= (len(traj['traj']) - i) ** length_penalty
            all_returns.append(node.return_to_go)


def standardize_returns(data):
    all_returns = [] 
    for traj in data:
        for i, node in enumerate(traj['traj']):
            all_returns.append(node.return_to_go)
    
    ## compute mean and std of rewards
    mean_reward = np.mean(all_returns)
    std_reward = np.std(all_returns)
    ## normalize to zero mean and unit variance
    for traj in data:
        for node in traj['traj']:
            node.return_to_go = (node.return_to_go - mean_reward) / std_reward


class ReasoningTrajectoryDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 bin_boundaries=None,
                 intermediate_bin_boundaries=None,
                 evaluator=None,
                 end_of_traj_str=' ##',
                 end_of_action_str=' |'
                 ):
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.reward_precision = config.get('reward_precision', 2)
        self.max_length = config.max_length
        self.loss_over_actions = True
        self.end_of_action_str = end_of_action_str
        self.end_of_traj_str = end_of_traj_str
        self.predict_action_only = False #### TRUE is not implemented correctly
        self.evaluator = evaluator
        
        self.data = self._load_data()

        reward_source = config.get('reward_source', 'R')
        assert reward_source in ['R', 'Q', 'uniform', 'random', 'visit_count', 'R_visit_count'], 'reward_source must be either R, Q, uniform, random, visit_count, R_visit_count'

        if reward_source == 'Q':
            print("Using Q values as rewards!!!")
            ### replace node.reward with node.Q
            for traj in self.data:
                for node in traj['traj']:
                    node.reward = node.Q

        elif reward_source == 'uniform':
            print("Using uniform rewards!!!")
            for traj in self.data:
                for node in traj['traj']:
                    node.reward = 1.0

        elif reward_source == 'random':
            print("Using random rewards!!!")
            for traj in self.data:
                for node in traj['traj']:
                    node.reward = np.random.uniform(0, 1)
        
        elif reward_source == 'visit_count':
            print("Using visit count as rewards!!!")
            for traj in self.data:
                for node in traj['traj']:
                    node.reward = len(node.cum_rewards)

        elif reward_source == 'R_visit_count':
            #### multiply R with visit count
            print("Using R * visit count as rewards!!!")
            for traj in self.data:
                for node in traj['traj']:
                    node.reward = node.reward * len(node.cum_rewards)
        
        ### check if any reward is negative 
        if any(node.reward < 0 for traj in self.data for node in traj['traj']):
            print("Found negative rewards. Correcting by setting reward := exp(reward)")
            for traj in self.data:
                for node in traj['traj']:
                        node.reward = np.exp(node.reward)
        
        ## filter data
        if config.get("filter_data", False):
            print("Filtering data...")
            print("Old data size:", len(self.data))
            self.data = self.filter_data(self.data)

            print("New data size:", len(self.data))
        
        if config.get("calibrate_rewards_based_on_final_answer", False):
            print("Calibrating rewards based on final answer...")
            self.calibrate_rewards_based_on_final_answer(self.data, 
                                                        factor=config.reward_calibration_factor, 
                                                        calibration_method=config.reward_calibration_method)

        print("Computing returns to go for every node...")
        self.compute_returns_to_go(self.data)
        
        if config.get("normalize_rewards", False):
            print("Normalizing returns...")
            normalize_returns(self.data)

        if config.get("standardize_rewards", False):
            print("Standardizing returns...")
            standardize_returns(self.data)
        
        if config.get("balance_data", False):
            print("Balancing data...")
            self.data = self.balance_data(self.data, 
                                     n_bins=3, 
                                     method=config.balance_data_method)

        if not self.config.get('use_action_level_rewards', False):
            rets = self.get_returns_to_go(self.data)
        else:
            print("Using Action level rewards!")
            rets = self.get_rewards(self.data)
        
        if config.quantize_reward:
                ## bin all_returns into quantize_reward_bins bins
            if bin_boundaries is None:
                bin_boundaries = build_bin_boundaries(rets, 
                                                     method=config.quantization_method, 
                                            n_bins=config.quantize_reward_bins)

            self.bin_boundaries = bin_boundaries
            rets = [self.get_bin(r) for r in rets]
            ### create special token for each bin if it doesn't exist 
            if not any(BIN_TO_RETURN_TOKEN[b] in self.tokenizer.get_added_vocab() for b in range(1, self.config.quantize_reward_bins + 1)):
                self.tokenizer.add_tokens([BIN_TO_RETURN_TOKEN[b] for b in range(1, config.quantize_reward_bins + 1)])

        
        self.rets = rets
        self.max_return_to_go = max(rets)
        train_on_top_quantile = False

        if self.config.get('correct_only', False):
            assert self.config.quantize_reward_bins == 1, 'correct_only is only supported for quantize_reward_bins = 1'
            ### filter out incorrect trajectories
            len_before = len(self.data)
            print("**** TRAINING ON CORRECT TRAJECTORIES ONLY! ****")
            self.data = [traj for traj in tqdm(self.data, desc="filtering incorrect trajectories") if self.is_correct(traj)]
            print("Threw away {} incorrect trajectories".format(len_before - len(self.data)))
        
        data = []
        for traj in tqdm(self.data, desc='Constructing trajectories'):
            tokens = self.tokenize_traj(traj)
            if train_on_top_quantile and self.get_bin(tokens['returns'][0]) in [self.config.quantize_reward_bins, self.config.quantize_reward_bins - 1]:
                data.append(tokens)
            elif not train_on_top_quantile:
                data.append(tokens)

        if self.config.get("valid_only", False):
            print("Filtering out invalid trajectories...")
            print("Old data size:", len(data))
            new_data = [] 
            for traj in tqdm(data, desc='filtering invalid trajectories'):
                if self.is_valid(traj):
                    new_data.append(traj)
            
            data = new_data
            print("New data size:", len(data))

        ## print maximum length of the trajectories
        max_length = max(len(d['token_ids']) for d in data)
        print("Maximum length of the trajectories: {}".format(max_length))

        self.data = data
        print("Processed data from {}".format(self.data_path))
        print("Dataset size: {}".format(len(self.data)))
        print("Number of steps: {}".format(sum(len(d['returns']) for d in self.data)))
        ## print reward stats 
        print('Returns-to-go Stats:')
        print(f'Min: {min(rets)}')
        print(f'Max: {max(rets)}')
        print(f'Mean: {np.mean(rets)}')

        if config.quantize_reward:
            ## print the percentage of each bin
            print('Percentage of each bin:')
            for i in range(1, config.quantize_reward_bins + 1):
                print(f'Bin {i}: {rets.count(i) / len(rets)}')
        


    def get_bin(self, reward):
        if self.bin_boundaries is None:
            return reward
        bin = np.digitize(reward, self.bin_boundaries)
        return min(bin, self.config.quantize_reward_bins)
            
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
                            print("Error loading {}".format(os.path.join(root, file)))
                            continue
                        all_trajs.extend(_)
                        if self.config.get('debug', False):
                            break

        assert len(all_trajs) > 0, 'No data found in {}'.format(self.data_path)

        if 'trajs' in all_trajs[0]:
            ## flatten out data to {question, answer, traj}
            data = []
            for traj in all_trajs:
                for t in traj['trajs']:
                    data.append({
                        'example': traj['example'],
                        'traj': t
                    })
        else:
            data = all_trajs

        ## filter out repeated trajectories
        print("Filtering out duplicates...")
        print("Old data size:", len(data))
        init_goal_traj = set()
        new_data = []
        for traj in data:
            traj_id = (traj['example']['init'], traj['example']['goal'], "\n".join([node.action for node in traj['traj']]))
            if traj_id not in init_goal_traj:
                new_data.append(traj)
                init_goal_traj.add(traj_id)

        data = new_data
        print("New data size:", len(data))

        #### reconstuct nodes to avoid weird reference issues
        new_data = []
        for traj in data:
            new_traj = []
            for node in traj['traj']:
                new_node = Node(
                    action=node.action,
                    reward=node.reward,
                    state=node.state,
                    cum_rewards=node.cum_rewards,
                    Q=node.Q,
                    return_to_go=None
                )
                new_traj.append(new_node)
            new_data.append({
                'example': traj['example'],
                'traj': new_traj
            })

        data = new_data
        return data
    
    def tokenize_traj(self, traj):
        '''
        returns tokenization
          QUESTION: <question> [SEP] REWARD STATE ACTION REWARD STATE ACTION....
        '''
        init = traj['example']['init']
        goal = traj['example']['goal']
        traj_str = '[init]\n{}\n[goal]\n{}\n'.format(init, goal)
        traj_token_ids = self.tokenizer.encode(traj_str, add_special_tokens=False)
        traj_loss_mask = [0] * len(traj_token_ids)
        returns = []

        if self.config.get('max_traj_steps', None):
            traj['traj'] = traj['traj'][:self.config.max_traj_steps]

        #### validate returns 
        assert all(node.reward > 0 for node in traj['traj']), f'negative reward in {traj}'

        for i, node in enumerate(traj['traj']):
            return_to_go = node.return_to_go
            
            ### RETURN TO GO ####
            if self.config.quantize_reward:
                bin = self.get_bin(return_to_go)
                ## get special token for this bin
                token_str = '<RET{}>\n' if i == 0 else ('<ret{}>\n' if self.config.get('use_intermediate_tokens', True) else '')
                s = token_str.format(bin)
            else:
                s = '[ret]\n'
                return_to_go = round(node.return_to_go, self.reward_precision)
                s+= f'{return_to_go}\n'

            if s.strip():
                tokens = self.tokenizer.encode(
                    s,
                    add_special_tokens=False
                )
                traj_str += s
                traj_token_ids.extend(tokens)
                mask = [0 if (self.config.quantize_reward and i == 0) else 1] * len(tokens)

                traj_loss_mask.extend(mask)
                
            ######## ACTION AND STATE ########        
            s= f'[action]\n{node.action}\n'
            #if not self.config.quantize_reward: ### do not predict action reward in the case of quantization
            #    s+=f'[reward]\n{node.reward}\n' 
            s+= f'[state]\n{node.state.blocks_state}\n'

            if i == len(traj['traj']) - 1:
                s += self.end_of_traj_str.strip()
            
            tokens = self.tokenizer.encode(
                s,
                add_special_tokens=False
            )
            traj_str += s
            traj_token_ids.extend(tokens)
            mask = [1] * len(tokens)

            traj_loss_mask.extend(mask)
            returns.append(return_to_go)


        assert len(traj_token_ids) == len(traj_loss_mask)

        if len(traj_token_ids) > self.max_length:
            traj_token_ids = traj_token_ids[:self.max_length]
            traj_loss_mask = traj_loss_mask[:self.max_length]
        
        ret_dict = {
            'traj_str': traj_str,
            'token_ids': traj_token_ids,
            'loss_mask': traj_loss_mask,
            'returns': returns,
            'example': traj['example'],
        }
        if 'is_correct' in traj:
            ret_dict['is_correct'] = traj['is_correct']
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

    def collate_fn(self, batch):
        input_ids = pad_sequence([b['input_ids'] for b in batch], batch_first=True)
        attention_mask = pad_sequence([b['attention_mask'] for b in batch], batch_first=True)
        labels = pad_sequence([b['labels'] for b in batch], batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def filter_data(self, data):
        ## remove trajectories with 1 step or less or more than 12 steps
        new_data = []
        for traj in data:
            n_steps = len(traj['traj'])
            if n_steps <= 1 or n_steps > 12:
                continue
            new_data.append(traj)

        return new_data
        
    def get_returns_to_go(self, data):
        ### sum returns to go
        rets = []
        for traj in data:
            for i, node in enumerate(traj['traj']):
                if self.config.get('use_intermediate_tokens', True) or i == 0:
                    rets.append(node.return_to_go)
                else:
                    break
        return rets

    def balance_data(self, data, n_bins=10, method="upsample"):
        ### bins data into K bins. 
        ### balace the data by up-sampling the bins with less data
        print("Data size before balancing:", len(data))

        returns_to_go = self.get_returns_to_go(data)
        bin_boundaries = build_bin_boundaries(returns_to_go, method='histogram', n_bins=n_bins)

        bin_counts = [0] * n_bins
        bin_id_to_traj = [[] for _ in range(n_bins)]

        for traj in data:
            return_to_go = traj['traj'][0].return_to_go
            bin = np.digitize(return_to_go, bin_boundaries)# 
            bin = min(bin, n_bins)

            assert bin >= 1 and bin <= n_bins, f'bin {bin} out of range'
            bin_counts[bin - 1] += 1
            bin_id_to_traj[bin - 1].append(traj)

        max_bin_count = max(bin_counts)
        min_bin_count = min(bin_counts)
        new_data = []
        
        if method == "upsample": ## upsample the bins with less data
            for i, c in enumerate(bin_counts):
                if c == max_bin_count:
                    new_data.extend(bin_id_to_traj[i])
                else:
                    new_data.extend(np.random.choice(bin_id_to_traj[i], max_bin_count, replace=True))
            
        elif method == "downsample":
        ### downsample 
            for i, c in enumerate(bin_counts):
                if c == min_bin_count:
                    new_data.extend(bin_id_to_traj[i])
                else:
                    new_data.extend(np.random.choice(bin_id_to_traj[i], min_bin_count, replace=False))
        else:
            raise ValueError('method must be either upsample or downsample')

        print("Data size after balancing:", len(new_data))
        #import ipdb; ipdb.set_trace()
        return new_data

    def get_intermediate_returns(self, data):
        #### returns to go in the middle of the trajectories
        returns = []
        for traj in data:
            traj_returns = [node.return_to_go for node in traj['traj']]
            returns.extend(traj_returns[1:])
        return returns

    def compute_returns_to_go(self, data):
        ### sum returns to go
        for traj_idx, traj in enumerate(data):
            return_to_go = sum([node.reward for node in traj['traj']])
            for i, _ in enumerate(traj['traj']):
                traj['traj'][i].return_to_go = return_to_go
                return_to_go -= traj['traj'][i].reward

    def is_correct(self, traj):
        if 'is_correct' in traj:
            return traj['is_correct']
        action_trace = '\n'.join([node.action for node in traj['traj']])
        return self.evaluator.eval_output(answer=traj['example'], output=action_trace)
    
    def calibrate_rewards_based_on_final_answer(self, data, calibration_method='scale', factor=10.0):
        ### caluclate mean and std 
        all_rewards = []
        for traj in data:
            for node in traj['traj']:
                all_rewards.append(node.reward)
        
        std_reward = np.std(all_rewards)
        for traj in tqdm(data, desc="Calibrating rewards based on final answer"):
            if self.is_correct(traj):
                for node in traj['traj']:
                    if calibration_method == 'scale':
                        assert node.reward > 0, 'reward must be positive'
                        node.reward *= factor
                    elif calibration_method == 'add_std':
                        node.reward += std_reward * factor
                    else:
                        raise ValueError('calibration_method must be either scale or add_std')

    def get_rewards(self, data):
        rewards = []
        for traj in data:
            for node in traj['traj']:
                rewards.append(node.reward)
        return rewards

    def is_valid(self, traj):
        return True