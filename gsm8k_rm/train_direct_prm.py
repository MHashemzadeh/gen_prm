## train a process reward model over the MCTS trajectories...
import os, sys
## turn off FutureWarnings
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import warnings
from utils.prm_dataset import MathShepherdDataset, GenerativeDirectPRMDataset, PRMCoTEvalDataset
import numpy as np
import wandb
import torch
import json
from transformers import pipeline
from utils.config import Config
from process_reward_model import FewshotCoTProcessRewardModel
from tqdm import tqdm
from copy import deepcopy
from datasets import load_metric

class PRMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.generation_pipeline = kwargs.pop('generation_pipeline')
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only=False,
                        ignore_keys=None):
        _inputs = self._prepare_inputs(inputs)
        
        input_texts = _inputs['input_text']
        self.tokenizer.padding_side = 'left'
        tokenized = self.tokenizer(input_texts, padding=True, truncation=False, return_tensors='pt', add_special_tokens=False)
        
        # Move inputs to same device as model
        tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
        
        self.tokenizer.padding_side = 'right'
        
        with torch.no_grad():
            generated_outputs = model.generate(
                tokenized['input_ids'],
                attention_mask=tokenized['attention_mask'],
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        ## remove the input part from the generated outputs
        input_length = tokenized['input_ids'].shape[1]
        generated_outputs = generated_outputs[:, input_length:]
        generated_texts = self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        gt_step_labels = _inputs['step_labels']
                
        # Process generated texts to extract step labels
        flattened_gt_labels = [] 
        flattened_generated_labels = []

        for i, text in enumerate(generated_texts):
            gt_step_label = [k for k in gt_step_labels[i] if k in [0, 1]] # remove padding 
            analysis_lines = [line for line in text.split('\n') if 'Correct?' in line]
            if len(analysis_lines) != len(gt_step_label):
                print(f"Mismatch in number of steps between generated analysis {len(analysis_lines)} and ground truth {len(gt_step_label)}")
            else:
                print(">>>>>>> Model output:\n", text)

            for line, gt_label in zip(analysis_lines, gt_step_label):
                if 'yes' in line.strip().lower():
                    flattened_generated_labels.append(1)
                elif 'no' in line.strip().lower():
                    flattened_generated_labels.append(0)
                
                flattened_gt_labels.append(gt_label)
        
        assert len(flattened_generated_labels) == len(flattened_gt_labels)
        # loss is eror rate
        loss = -1 
        
        if len(flattened_generated_labels) > 0:
            loss = sum([1 for a, b in zip(flattened_generated_labels, flattened_gt_labels) if a != b]) / len(flattened_generated_labels)

        loss = torch.tensor(loss).to(self.model.device)

        if prediction_loss_only:
            return (loss, None, None)
        
        flattened_generated_labels = torch.tensor(flattened_generated_labels).to(self.model.device)
        flattened_gt_labels = torch.tensor(flattened_gt_labels).to(self.model.device)

        return loss, flattened_generated_labels, flattened_gt_labels
    
accuracy_metric = load_metric('accuracy')
f1_metric = load_metric('f1')
recall_metric = load_metric('recall')
precision_metric = load_metric('precision')

def compute_metrics(eval_pred):
    predictions_all, labels_all = eval_pred
    metrics = {}
    ### compute accuracy, f1, recall, precision
    metrics['accuracy'] = accuracy_metric.compute(predictions=predictions_all, references=labels_all)['accuracy']
    metrics['f1'] = f1_metric.compute(predictions=predictions_all, references=labels_all)['f1']
    metrics['recall'] = recall_metric.compute(predictions=predictions_all, references=labels_all)['recall']
    metrics['precision'] = precision_metric.compute(predictions=predictions_all, references=labels_all)['precision']
    metrics['n_eval'] = len(predictions_all)
    return metrics


def main(cfg: Config) -> None:
    warnings.filterwarnings(
        action='ignore',
        category=UserWarning,
        message=
        'torch.distributed.*_base is a private function and will be deprecated.*'
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
    tokenizer.model_max_length = cfg.data.max_length

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    ## save tokenizer 
    ## load and construct training dataset
    if isinstance(cfg.data.data_dir, list):
        data_path = [os.path.join(p, 'train') for p in cfg.data.data_dir]
    else:
        data_path = [os.path.join(cfg.data.data_dir, 'train')]


    if cfg.train.evaluation_strategy != 'no':
        _eval_dataset_gsm8k = MathShepherdDataset(
            tokenizer=tokenizer,
            config=cfg.data,
            split='dev',
            )
        
        _math_cfg = deepcopy(cfg.data)
        setattr(_math_cfg, 'task', 'math')
        
        _eval_dataset_math = MathShepherdDataset(
            tokenizer=tokenizer,
            config=_math_cfg,
            split='dev',
            )
        
        all_eval_data = _eval_dataset_gsm8k.data + _eval_dataset_math.data
        
        eval_dataset = PRMCoTEvalDataset(
            examples=all_eval_data,
            tokenizer=tokenizer,
            config=cfg.data,
            split='eval',
            )
        
        ## shuffle eval dataset
        np.random.shuffle(eval_dataset.data)
        eval_dataset.data = eval_dataset.data[:2000]

    dataset_class = GenerativeDirectPRMDataset
    train_dataset = dataset_class(
        data_path=data_path,
        tokenizer=tokenizer,
        config=cfg.data,
        split='train',
    )
    

    if cfg.train.evaluation_strategy != 'no':
        eval_questions = set([d.get('question', d.get('problem')) for d in eval_dataset.data])
        print("Filtering out dev questions from training data")
        old_train_size = len(train_dataset)
        ### keep only questions that are not in dev questions
        train_dataset.data = [d for d in train_dataset.data if d.get('question', d['problem']) not in eval_questions]
        print("Filtered out {} dev questions from training data".format(old_train_size - len(train_dataset)))
        print("Got {} dev examples".format(len(eval_dataset)))

        ## make sure no overlap between train and dev questions TODO

    print("Got {} training examples".format(len(train_dataset)))

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    ### init wandb 
    wandb.init(project='self-taught-prm', config=cfg.to_dict())

    cfg.train.output_dir = os.path.join(cfg.train.output_dir, wandb.run.name)
    ### make sure output_dir exists
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        output_attentions=False,
        output_hidden_states=False,
        torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float16,
        device_map=device_map,
    )
    
    model.config.pad_token_id = model.config.eos_token_id
    training_args = TrainingArguments(**cfg.train.to_dict(), remove_unused_columns=False, 
                                    metric_for_best_model='f1', 
                                    greater_is_better=True,
                                    load_best_model_at_end=cfg.train.evaluation_strategy != 'no')
    
    # Generate COT analysis using the model
    generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,  # Adjust as needed
        num_return_sequences=1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=train_dataset.collate_fn,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset if cfg.train.evaluation_strategy != 'no' else None,
        generation_pipeline=generation_pipeline,
    )

    trainer.train()
    ### save best model and tokenizer
    trainer.save_model(os.path.join(cfg.train.output_dir, 'best_model'))
    tokenizer.save_pretrained(os.path.join(cfg.train.output_dir, 'best_model'))
    
    # save config
    with open(os.path.join(cfg.train.output_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_prm.py <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    cfg = Config.load_config(config_path)
    main(cfg)
