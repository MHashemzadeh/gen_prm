import re
from typing import Optional, Union

from reasoners.base import AlgorithmOutput


def retrieve_answer(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, 'aggregated_result', None)) is not None:
            return result
        output = output.terminal_state
    
    if isinstance(output, list):
        if hasattr(output[-1], 'step'): # mcts
            output = output[-1].step
        elif hasattr(output[-1], 'terminal_node'): # beam search
            output = output[-1].terminal_node.action
    try:
        match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*\..*', output)
    except:
        return ''
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: Union[str, dict]) -> str:
    if isinstance(answer, dict):
        answer = answer['answer']
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer
