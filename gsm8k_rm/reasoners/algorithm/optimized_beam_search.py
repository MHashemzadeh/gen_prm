from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example
from typing import NamedTuple, List, Tuple, Callable, Any, Union, Optional
import numpy as np
import warnings
import random
from copy import deepcopy
import itertools

class BeamSearchNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, 
                 state: State, 
                 action: Action, 
                 reward: float, 
                 parent: Optional['BeamSearchNode'] = None, 
                 children: Optional[List['BeamSearchNode']] = None
                ) -> None:
        
        self.id = next(BeamSearchNode.id_iter)  
        self.state = state
        self.action = action
        self.reward = reward
        self.parent = parent
        self.children = children if children is not None else []

    def add_child(self, child: 'BeamSearchNode'):
        self.children.append(child)
    
    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path

class BeamSearchResult(NamedTuple):
    terminal_node: BeamSearchNode
    cum_reward: float
    tree: BeamSearchNode


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 beam_size: int, 
                 max_depth: int, 
                 sampling_strategy: str = 'argmax', # sampling strategy, argmax or softmax
                 replace: Optional[bool] = None, # whether to sample with replacement
                 temperature: Optional[float] = None, # temperature for softmax sampling
                 temperature_decay: Optional[float] = None, # temperature decay, default to no decay
                 reject_sample: Optional[bool] = None, # whether to reject the samples with reward less than the reject_min_reward
                 reject_min_reward: Optional[float] = None, # the minimum reward to reject the sample
                 unbiased: Optional[bool] = None, # whether to use unbiased sampling
                 reward_aggregator: Union[Callable[[List[Any]], float], str] = 'last', # how to aggregate the reward list
                 action_dedup: bool = True, # whether to deduplicate the actions
                 early_terminate: bool = True, # whether to add to terminal beam if the action is terminal
                 return_beam: bool = False # whether to return the beam instead of the best trace
                ) -> None:
        # Initialize the BeamSearch class
        self.beam_size = beam_size
        self.max_depth = max_depth
        self.sampling_strategy = sampling_strategy
        self.replace = replace
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.reject_sample = reject_sample
        self.reject_min_reward = reject_min_reward
        self.unbiased = unbiased
        self.reward_aggregator = reward_aggregator
        self.action_dedup = action_dedup
        self.early_terminate = early_terminate
        self.return_beam = return_beam

        # Initializing the reward_aggregator based on the provided argument
        self._initialize_reward_aggregator()

        # Post processing after initialization
        self._post_initialization()

    def _initialize_reward_aggregator(self):
        # how to aggregate the reward list
        if self.reward_aggregator == 'cumulative' or self.reward_aggregator == 'accumulative':
            self.reward_aggregator = lambda x: sum(x)
        elif self.reward_aggregator == 'mean' or self.reward_aggregator == 'average':
            self.reward_aggregator = lambda x: sum(x) / len(x)
        elif isinstance(self.reward_aggregator, str) and self.reward_aggregator.startswith('last'):
            self.reward_aggregator = lambda x: x[-1]
        else:
            # if the reward_aggregator is a string but not the above, raise error
            if isinstance(self.reward_aggregator, str):
                raise NotImplementedError(f"Reward aggregator {self.reward_aggregator} is not implemented.")
    
    def _post_initialization(self):
        # if the temperature is set to 0, then we force the sampling strategy to be argmax
        if self.temperature and self.temperature < 1e-4:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Temperature is set to 0, sampling strategy is forced to be argmax.")

        # argmax = greedy = deterministic = topk
        if self.sampling_strategy in ['greedy', 'deterministic', 'topk']:
            self.sampling_strategy = 'argmax'
        
        # if sampling strategy not in argmax or stochastic, just use argmax
        if self.sampling_strategy not in ['argmax', 'stochastic']:
            self.sampling_strategy = 'argmax'
            warnings.warn(f"Sampling strategy only supports argmax or stochastic, but got {self.sampling_strategy}. \
                            Sampling strategy is changed to argmax automatically.")
        
        # if early_terminate is set to False, we need to inform the user that we will return the beam instead of the best trace
        if not self.early_terminate:
            self.return_beam = True
            warnings.warn(f"early_terminate is set to False, BeamSearch will return the beam instead of the best trace.")

    
    @staticmethod
    def softmax(x: List[float], temperature: float, unbiased: bool = False, action_probs: Optional[List[float]] = None) -> List[float]:
        e_x = np.exp(np.array(x) / temperature)

        if unbiased and action_probs is not None:
            # adjust the values by the action_probs
            adjusted_values = [ n*p for n, p in zip(e_x, action_probs)]

            return [p / sum(adjusted_values) / max(1, len(adjusted_values)) for p in e_x]

        return list(e_x / e_x.sum())


    def _sample(self, beam):
        # sort the beam by reward
        beam.sort(key=lambda x: x[2], reverse=True)
        if self.reject_sample:
            # reject the samples with reward less than the reject_min_reward
            beam = [x for x in beam if x[2] >= self.reject_min_reward]
        # return the top k
        return beam[:self.beam_size]

        

    def __call__(self, world: WorldModel[State, Action, State], config: SearchConfig[State, Action, State]):
        init_state = world.init_state()
        # root node
        root_node = BeamSearchNode(state=init_state, action=None, reward=0.0)
        # Initialize current beam with initial state
        cur_beam = [(root_node, [], 0.0)] # (node, reward_list, cum_reward)
        terminal_beam = []

        for depth in range(self.max_depth):
            new_beam = []

            # Collect states from current beam
            states = [item[0].state for item in cur_beam]
                        # Get actions for all states in batch
            batch_actions = config.get_actions_batch(states)

            # Process each beam item with its corresponding actions
            for beam_idx, beam_item in enumerate(cur_beam):
                node, reward_list, _ = beam_item[:3]
                state = node.state
                
                if self.early_terminate and world.is_terminal(state):
                    terminal_beam.append(beam_item)
                    continue

                actions = batch_actions[beam_idx]

                if self.action_dedup:
                    # only keep the unique actions
                    actions = list(set(actions))
                    if not actions:
                        continue

                # Prepare batch of state-action pairs
                state_action_pairs = [(state, action) for action in actions]
                
                # Get next states and rewards in batch
                next_states_and_aux = [world.step(sa[0], sa[1]) for sa in state_action_pairs]
                next_states = [nsa[0] for nsa in next_states_and_aux]
                aux_list = [nsa[1] for nsa in next_states_and_aux]
                
                # Get rewards in batch
                rewards = config.reward_batch(states=[state]*len(actions), actions=actions, aux_list=aux_list)
                # Process each action result
                for action_idx, (action, next_state, reward) in enumerate(zip(actions, next_states, rewards)):
                    # if the reward is a tuple, then it is (reward, aux)
                    if isinstance(reward, tuple):
                        reward, reward_aux = reward

                    # Add new reward to list of rewards
                    new_reward_list = reward_list + [reward]

                    # Compute new reward
                    new_reward = self.reward_aggregator(new_reward_list)

                    # Create new node
                    new_node = BeamSearchNode(state=next_state, action=action, reward=reward, parent=node)

                    # Add new node to children of current node
                    node.add_child(new_node)
                    new_beam.append((new_node, new_reward_list, new_reward))

                # check whether this is max_depth
                if depth == self.max_depth - 1:
                    terminal_beam.append(beam_item)

            if not new_beam: ### nothing to expand -- all terminal
                self.early_terminate = True
                break 
            
            # Sort new beam by reward
            new_beam.sort(key=lambda x: x[2], reverse=True)
            # Sample from new beam
            cur_beam = self._sample(new_beam)

            # Decay the temperature
            if self.temperature_decay:
                self.temperature *= self.temperature_decay
        
        if not self.early_terminate:
            # add the cur_beam to terminal_beam
            terminal_beam += cur_beam

        # Sort terminal beam by reward
        terminal_beam.sort(key=lambda x: x[2], reverse=True)

        if self.return_beam:
            # convert terminal_beam to a list of BeamSearchResult
            terminal_beam = [BeamSearchResult(
                                terminal_node=item[0],
                                cum_reward=item[2],  # Use the precomputed cum_reward
                                tree=root_node
                                ) for item in terminal_beam]
            
            return terminal_beam


        best_result = terminal_beam[0]
        result = BeamSearchResult(
            terminal_node=best_result[0],
            cum_reward=best_result[2],  # Use the precomputed cum_reward
            tree=root_node
            )

        return result
