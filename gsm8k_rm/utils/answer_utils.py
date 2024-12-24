import re
from typing import Optional
from collections import Counter


#GAME24_PROMPT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
#Obtain 24 using the numbers: {problem}

### Response:
#"""

GAME24_PROMPT = """Obtain 24 using the numbers: {problem}
 """

GAME24_INSTRUCTION_PROMPT = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate a solution to obtain 24 using the given numbers. The solution should be in one of the following types: Correct, Close, Not Close.

Correct: The solution is correct and leads to the number 24.
Close: The solution is incorrect but close to the number 24.
Not Close: The solution is incorrect and not close to the number 24.

Problem: {problem}
Type: {type}

### Response:
"""

GAME24_INSTRUCTION_SCORE_PROMPT = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate a solution to obtain 24 using the given numbers. The quality of the solution you generate should depend on a score I will give you. The higher the score the more likely the solution to lead to 24. The highest score is {max_score} and when provided, you should generate a correct solution that leads to 24.

Problem: {problem}
Score: {score}

### Response:
"""

GAME24_INSTRUCTION_TREE_PROMPT = """"Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Find a correct solution to use the given numbers and basic arithmetic operations (+ - * /) to obtain 24, You should keep trying multiple solutions until you find the correct one. 

Numbers: {problem}

### Response:
"""


## enumerable reward tokens 
BIN_TO_RETURN_TOKEN = {
    k:v for k,v in enumerate(['<RET{}>'.format(i) for i in range(32)])
}

BIN_TO_INTERMEDIATE_TOKEN = {
    k:v for k,v in enumerate(['<ret{}>'.format(i) for i in range(32)])
}


VALUE_TO_RETURN_STR = lambda r: 'R: {:.0f}'.format(r)

ACTION_SEP = '\n '## space is very important 

def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except Exception as e:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except Exception as e:
        pass
    return output == answer


def get_majoirty_answer(answers):
    if len(answers) == 0: # all solutions are invalid -- do not have a final answer
        return '[INVALID]'
    else:
        ## pick the majority answer
        voted_answer = Counter(answers).most_common(1)[0][0]
        ## pick the first solution that has the voted answer
        return voted_answer
    

def get_action_trace_from_plan_str(plan_str):
    ### return all lines after lines with [ACTION] 
    lines = plan_str.split('\n')
    action_trace = []
    for i, line in enumerate(lines):
        if '[action]' in line and i + 1 < len(lines):
            action_trace.append(lines[i+1])
    
    return "\n".join(action_trace)


def extract_step_cots_and_labels(output_str):
    analysis_lines = [line for line in output_str.split('\n') if 'Correct?' in line]
    step_cots = []
    step_labels = []

    for line in analysis_lines:
        parts = line.split('Correct?')
        cot = 'Correct?'.join(parts[:-1])  # Join all parts except the last one
        output = parts[-1]  # The last part after the final 'Correct?'
        if 'yes' in output.strip().lower():
            step_labels.append(1)
            step_cots.append(cot.strip())
        elif 'no' in output.strip().lower():
            step_labels.append(0)
            step_cots.append(cot.strip())
            
    return step_cots, step_labels