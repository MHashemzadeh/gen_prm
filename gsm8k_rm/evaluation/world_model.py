import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils
from reasoners.base import Example


class SubResult(NamedTuple):
    step: str
    confidence: float


GSM8kState = list[SubResult]
GSM8kAction = str
GSM8kExample = str


class GSM8kPromptDict(TypedDict):
    instruction: str
    interactive_examples: list[str]
    useful_examples: list[str]
    question_prefix: str
    subquestion_prefix: str
    overall_question_prefix: str
    answer_prefix: str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 n_confidence=8,
                 batch_size=2,
                 temperature=0.8,
                 early_stop_base=None,
                 early_stop_threshold=1.) -> None:
        super().__init__()
        self.base_model = base_model
        self.batch_size = batch_size
        self.n_confidence = n_confidence
        self.temperature = temperature
        self.early_stop_base = early_stop_base if early_stop_base is not None else n_confidence
        self.early_stop_threshold = early_stop_threshold
        self.prompt_examples = ""
        self.n_shots = 0

    def update_example(self, example: Example, prompt: GSM8kPromptDict = None) -> None:
        super().update_example(example, prompt)
        assert prompt is not None
        self.prompt = prompt
        with io.StringIO() as f:
            f.write(self.prompt['instruction'] + '\n\n')
            for idx, example in enumerate(self.prompt['interactive_examples']):
                f.write(example.format(idx=idx + 1) + '\n\n')
            self.n_shots = len(self.prompt['interactive_examples'])
            self.prompt_examples = f.getvalue()

    def init_state(self) -> list:
        return []

    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()
        state.append(SubResult(action, 1.0))
        #print(state)
        aux = {'confidence': 1.0}
        return state, aux

    def is_terminal(self, state: GSM8kState) -> bool:
        if len(state) > 0 and ("the answer is" in state[-1].step.lower() or "final answer" in state[-1].step.lower()):
            return True
        else:
            return False
