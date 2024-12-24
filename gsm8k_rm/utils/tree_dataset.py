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


class SearchTreeDataset(Dataset):
    def __init__(self, 
                 data_path: str, 
                 tokenizer, 
                 config=None, 
                 evaluator=None,
                 end_of_traj_str=' ##',
                 end_of_action_str=' |',
                 **kwargs):
        
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.end_of_action_str = end_of_action_str
        self.end_of_traj_str = end_of_traj_str
        self.predict_action_only = False #### TRUE is not implemented correctly
        self.evaluator = evaluator

        default_config = {
            'n_samples_per_problem': 10,
            'n_trajs_per_prompt': 3,
            'traj_sampling_strategy': 'fixed-incremental',
        }

        for k in default_config:
            if k not in config:
                config[k] = default_config[k]

        self.data = self._load_data()

        ### make make sure they are all positive. 
        ### check if any reward is negative 
        if any(node.reward < 0 for traj in self.data for node in traj['traj']):
            print("Found negative rewards. Correcting by setting reward := exp(reward)")
            for traj in self.data:
                for node in traj['traj']:
                        node.reward = np.exp(node.reward)
        
        print("Computing returns to go for every node...")
        self.compute_returns_to_go(self.data)
        
        if config.get("normalize_rewards", False):
            print("Normalizing returns...")
            self.normalize_returns(self.data)

        rets = self.get_returns_to_go(self.data)
        rewards = self.get_rewards(self.data)

        ##### set total_return for each trajectory
        for traj in self.data:
            traj['total_return'] = traj['traj'][0].return_to_go
        
        if config.quantize_reward:
            ## bin all_returns into quantize_reward_bins bins
            bin_boundaries = self.build_bin_boundaries(rets, 
                                                    method=config.quantization_method, 
                                        n_bins=config.quantize_reward_bins)

            ##### update returns to go for each traj in data 
            for traj in self.data:
                traj['total_return'] = self.get_bin(traj['total_return'], bin_boundaries)

        #### bin all rewards into 3 bins: unlikely, likely, very likely
        bin_boundaries = self.build_bin_boundaries(rewards,
                                                method=config.quantization_method, 
                                    n_bins=3) # TODO make this configurable

        for traj in self.data:
            for node in traj['traj']:
                node.reward = self.get_bin(node.reward, bin_boundaries)
        
        self.rets = rets
        self.max_return_to_go = max(rets)

        self.data = self.construct_prompts(self.data)
       
        print("Processed data from {}".format(self.data_path))
        print("Dataset size: {}".format(len(self.data)))

    def get_bin(self, reward, bin_boundaries=None):
        bin = np.digitize(reward, bin_boundaries)
        bin = min(bin, len(bin_boundaries) - 1)
        assert bin >= 1 and bin < len(bin_boundaries), f"bin: {bin}, reward: {reward}, boundaries: {bin_boundaries}"
        return bin
            
    def _load_data(self):
        raise NotImplementedError
    
    def construct_prompts(self, traj):
       raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item['token_ids'])
        attention_mask = torch.tensor([1] * len(item['token_ids']))
        #loss_mask = torch.tensor(item['loss_mask'])
        labels = input_ids.clone()
        
        #labels[loss_mask == 0] = -100

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

    def get_returns_to_go(self, data):
        ### sum returns to go
        rets = []
        for traj in data:
            rets.append(traj['traj'][0].return_to_go)
        return rets

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
    
    def is_valid(self, traj):
        raise NotImplementedError

    def normalize_returns(self, data, length_penalty=1.0):
        #### normalize by length then standarsize
        all_returns = [] 
        for traj in data:
            for i, node in enumerate(traj['traj']):
                ## normalize by number of remaining steps 
                node.return_to_go /= (len(traj['traj']) - i) ** length_penalty
                all_returns.append(node.return_to_go)


    def build_bin_boundaries(self, values, method='histogram', n_bins=10):
        if method == 'histogram':
            bin_boundaries = torch.linspace(min(values), max(values), n_bins + 1)
        elif method == 'percentile':
            bin_boundaries = torch.tensor(np.percentile(values, np.linspace(0, 100, n_bins + 1)))
        else:
            raise ValueError('quantize_reward_method must be either histogram or percentile')
        
        assert len(bin_boundaries) == n_bins + 1
        return bin_boundaries
    
    def get_rewards(self, data):
        rewards = []
        for traj in data:
            for node in traj['traj']:
                rewards.append(node.reward)
        return rewards