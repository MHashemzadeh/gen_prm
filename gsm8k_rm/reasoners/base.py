from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional, runtime_checkable, Tuple
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList
from datetime import datetime
import os, sys, pickle
from tqdm import tqdm
import torch
import jsonlines
State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: list[np.ndarray] = None


class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(self,
                          prefix: str,
                          contents: list[str],
                          **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...


class WorldModel(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> Union[State, Tuple[State, dict]]:
        """ Returns the next state and optionally an auxiliary data dict

        :param state: The current state
        :param action: The action to take
        :return: The next state and optionally an auxiliary data dict
        """
        ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example, prompt = None) -> None:        
        if prompt is not None:
            self.prompt = prompt
        self.example = example


class SearchConfig(ABC, Generic[State, Action, Example]):
    def __init__(self) -> None:
        self.example = None
        self.prompt = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    @abstractmethod
    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example, prompt = None) -> None:
        if prompt is not None:
            self.prompt = prompt
        self.example = example


@runtime_checkable
class AlgorithmOutput(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> AlgorithmOutput: ...


class Reasoner(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Example, prompt = None, **kwargs) -> AlgorithmOutput[State]:
        self.world_model.update_example(example, prompt=prompt)
        self.search_config.update_example(example, prompt=prompt)
        return self.search_algo(self.world_model, self.search_config, **kwargs)

class Evaluator():
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def sample_prompt(self,
                      shuffle_prompt,
                      num_shot,
                      sample_prompt_type):
        pass
    
    def evaluate(self,
                 reasoner,
                 shuffle_prompt=True,
                 num_shot=4,
                 resume=0,
                 log_dir=None):

        self.dataset = list(self.full_dataset)[resume:]
        
        if hasattr(self, 'num_workers') and self.num_workers > 1:
            self.dataset = self.dataset[self.worker_id::self.num_workers]

        print(f"Running on: {len(self.dataset)} examples")

        try:
            algo_name = reasoner.search_algo.__class__.__name__
        except:
            algo_name = "unknown"

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if log_dir is None:
                log_dir = f'logs/{self._dataset_name}_'\
                        f'{algo_name}/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                print(sys.argv, file=f)

        correct_count = 0

        all_trajs = []
        predictions = []
        search_traces = []
        for i, example in enumerate(tqdm(self.dataset,
                                            total=resume + len(self.dataset),
                                            initial=resume,
                                            desc=self._dataset_name,
                                            disable=self.disable_tqdm)):
            
            algo_output = reasoner(self.input_processor(example),
                                    prompt=self.sample_prompt(
                                        shuffle_prompt=shuffle_prompt,
                                        num_shot=num_shot,
                                        sample_prompt_type=self.sample_prompt_type))

            if self.sample_trajectories:
                trajs = extract_trajectories(reasoner.search_algo)
                all_trajs.append({
                    'example': example,
                    'trajs': trajs
                })
                
                search_traces.append({
                    'example': example,
                    'traces': extract_search_traces(algo_output)
                })

            output = self.output_extractor(algo_output)
            answer = self.answer_extractor(example)
            correct = self.eval_output(answer, output)
            predictions.append(int(correct))
            
            correct_count += correct
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=};'\
                        f'{accuracy=:.3f} ({correct_count}/{i + 1})'
            tqdm.write(log_str)

            if (not self.disable_log) and \
                (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
                with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                    print(log_str, file=f)
            
        if self.sample_trajectories:
            return accuracy, all_trajs
        
        ## create log dir if not exists
        os.makedirs(log_dir, exist_ok=True)
        ## log predictions
        if hasattr(self, 'worker_id') and self.sample_trajectories:
            with jsonlines.open(os.path.join(log_dir, 'predictions_worker_{}.jsonl'.format(self.worker_id)), 'w') as writer:
                for prediction in predictions:
                    writer.write(prediction)

        return accuracy

    @abstractmethod
    def eval_output(self, answer, output):
        pass


def _dfs(path, result):
    cur = path[-1]
    if cur.is_terminal or cur.children is None:
        result.append(path)
        return
    
    children_to_visit = [x for x in cur.children if x.state is not None]
    if len(children_to_visit) == 0:
        result.append(path)
        return
    
    for child in children_to_visit:
        _dfs(path + [child], result)

def extract_trajectories(search_algo_object):
    '''
    runs a DFS search over the MCTS tree and returns all paths from root to a terminal node. 
    '''
    if search_algo_object.root is None:
        return []
    result = []
    _dfs([search_algo_object.root], result)
    ## remove empty paths from result 
    result = [x for x in result if len(x) > 0]
    ## remove root node from all paths
    result = [x[1:] for x in result]

    return result

def extract_search_traces(search_output):
    '''
    runs a DFS search over the MCTS tree and returns all paths from root to a terminal node. 
    '''
    search_trace_roots = [node for node in search_output.tree_state_after_each_iter]
    search_traces = []
    for root in search_trace_roots:
        trace = []
        _dfs([root], trace)
        search_traces.append(trace)     
    
    return search_traces
