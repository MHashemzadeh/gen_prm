
from utils.prm_dataset import PRMTrajectoryDataset
import re
from collections import defaultdict
import json
import datasets


class PRM800KDataset(PRMTrajectoryDataset):
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
        data = [json.loads(line) for line in open('data/prm800k/phase2_train.jsonl', 'r')]
        return data
    
    def process_data(self, data):
        #gold_data = datasets.load('gsm8k', split='train')
        processed_data = []
        for d in data: 
            question = d['question']['problem']
            task = 'math'  # Since this appears to be a math problem based on the format
            
            # Get the trajectory steps from the label
            steps = []
            for step_data in d['label']['steps']:
                # Get the first completion's text and rating
                step = step_data['completions'][0]['text']
                rating = step_data['completions'][0]['rating']

                if '\n\n# Answer\n\n' in step:
                    step = step.replace('\n\n# Answer\n\n', ' The answer is $').strip() + '$'

                steps.append({'step': step, 'label': 1 if rating in [1, 0] else 0}) # correct/neutral is correct

            # Build trajectory string
            traj = self.step_sep.join([s['step'] for s in steps])
            # Extract step labels
            steps_and_labels = [{'step': s['step'], 'label': s['label']} for s in steps]
            
            # Check if trajectory is fully correct (all steps rated 1)
            is_traj_correct = all(s['label'] == 1 for s in steps)
             
            # Get ground truth solution
            gold_chain = d['question']['ground_truth_answer']

            ### some items in prm800k don't have a gold solution for some reason. Skip these.
            if any(s is None for s in [question, gold_chain]):
                continue
            
            processed_data.append({
                'question': question,
                'traj': traj,
                'steps_with_labels': steps_and_labels,
                'solution_label': 1 if is_traj_correct else 0,
                'answer': gold_chain,
                'task': task
            })

        return processed_data
        
    def clean_trajectory(self, traj):
        return traj


if __name__ == '__main__':
    import yaml
    from utils.config import Config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
    config = Config.load_config('conf/prm/paper/all/ovm.yaml')
    dataset = PRM800KDataset(tokenizer=tokenizer, config=config.data, split='train')
