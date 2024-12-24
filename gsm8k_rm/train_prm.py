## train a process reward model over the MCTS trajectories...
import os, sys
## turn off FutureWarnings
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
import warnings
from utils.prm_dataset import PRMPairwiseDataset, PRMBinaryDataset, MathShepherdDataset, PRMContinuosScoreDataset, FutureSamplingDataset, GenerativeInterleavedPRMDataset
import numpy as np
import wandb
from datasets import load_metric
from copy import deepcopy
import torch
from torch import nn
import subprocess
import yaml
import json

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return Config(config_dict)


class PRMCustomTrainer(Trainer):
    def __init__(self, *args, pos_label_token_id=None, neg_label_token_id=None, 
                 label_format='binary', **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_label_token_id = pos_label_token_id
        self.neg_label_token_id = neg_label_token_id
        self.label_format = label_format

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_format == 'binary':
            return self.compute_loss_binary(model, inputs, return_outputs)
        elif self.label_format == 'continuous':
            return self.compute_loss_continuous(model, inputs, return_outputs)
        else:
            raise ValueError("Invalid label format: {}".format(self.label_format))
    
    def compute_loss_continuous(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels'].float()
        loss_mask = inputs['loss_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(reduction='mean')

        ### compute loss between logits that are not masked
        logits_not_masked = logits.view(-1, logits.size(-1))[loss_mask.view(-1) == 1]
        labels_not_masked = labels.view(-1)[loss_mask.view(-1) == 1]

        ## transform labels to logits-like tensor with all zeros except for the correct label
        _labels = torch.zeros_like(logits_not_masked)
        bl = torch.arange(logits_not_masked.size(0))
        _labels[bl, self.pos_label_token_id] = labels_not_masked
        _labels[bl, self.neg_label_token_id] = 1 - labels_not_masked

        loss = loss_fct(logits_not_masked, _labels)
        if loss < 0:
            import ipdb; ipdb.set_trace()
        return (loss, outputs) if return_outputs else loss
    
    def compute_loss_binary(self, model, inputs, return_outputs=False):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        loss_mask = inputs['loss_mask']
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(reduction='mean')

        ### compute loss between logits that are not masked
        logits_not_masked = logits.view(-1, logits.size(-1))[loss_mask.view(-1) == 1]
        labels_not_masked = labels.view(-1)[loss_mask.view(-1) == 1]

        loss = loss_fct(logits_not_masked, labels_not_masked)
        return (loss, outputs) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only=True,
                        ignore_keys=None):
        inputs = self._prepare_inputs(inputs)                
        if not torch.is_floating_point(inputs['labels']): ### whether the labels come as continuous or binary (math-shpherd)
            inputs['labels'] = (inputs['labels'] == self.pos_label_token_id).long()
        
        labels = inputs['labels']
        loss_mask = inputs['loss_mask']
        loss_mask_with_first_error_only = inputs['loss_mask_with_first_error_only']
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                logits = outputs.logits
                ## get logits where loss_mask is 1
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                loss_mask = loss_mask.view(-1)
                loss_mask_with_first_error_only = loss_mask_with_first_error_only.view(-1)

                logits_all = logits[loss_mask == 1]
                logits_with_first_error_only = logits[loss_mask_with_first_error_only == 1]

                ## only keep logits at self.pos_label_token_id and self.neg_label_token_id
                logits_all = logits_all[:, [self.neg_label_token_id, self.pos_label_token_id]]
                logits_with_first_error_only = logits_with_first_error_only[:, [self.neg_label_token_id, self.pos_label_token_id]]

                assert logits_all.size(-1) == 2
                assert logits_with_first_error_only.size(-1) == 2
                ## labels should be 0 if neg_label_token_id and 1 if pos_label_token_id
                labels_all = labels[loss_mask == 1]
                labels_with_first_error_only = labels[loss_mask_with_first_error_only == 1]


        if prediction_loss_only:
            return (loss, None, None)

        return (loss, (logits_all, labels_all), (logits_with_first_error_only, labels_with_first_error_only))
    

####### we can train the PRM usign one of the following strategies:
# 1. Using step-level DPO: reward could come from final answer or from the MCTS reward.  
# 2. Using OVM just as binary prediction
# 3. Using standard reward model Log(sigmoid(reward(correct) - reward(incorrect))

accuracy = load_metric('accuracy')
f1 = load_metric('f1')
recall = load_metric('recall')
precision = load_metric('precision')


def compute_metrics(eval_pred):
    (predictions_all, labels_all), (predictions_with_first_error_only, labels_with_first_error_only) = eval_pred
    metrics = {}
    for prefix, preds, labels in [('', predictions_all, labels_all), 
                                  ('TRACE_', predictions_with_first_error_only, labels_with_first_error_only)]:
        preds = np.argmax(preds, axis=1)
        labels = labels.astype(np.int32)
        
        metric_fns = {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision
        }
        
        for metric_name, metric_fn in metric_fns.items():
            if metric_name == 'accuracy':
                result = metric_fn.compute(predictions=preds, references=labels)
            else:
                result = metric_fn.compute(predictions=preds, references=labels, average='macro')
            metrics[f'{prefix}{metric_name}'] = result[metric_name]
    
    return metrics


class TrainEvalCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    def on_epoch_end(self, args, state, control, **kwargs):
        control_copy = deepcopy(control)
        train_subset = deepcopy(self._trainer.train_dataset)
        train_subset.data = train_subset.data[:2000]
        print(">>>> Evaluating on training data...")
        self._trainer.evaluate(eval_dataset=train_subset, metric_key_prefix="train")
        return control_copy

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

    data_cls = None 
    if cfg.data.format == 'pairwise_stepwise':
        data_cls = PRMPairwiseDataset
    elif cfg.data.format == 'binary':
        data_cls = PRMBinaryDataset
    elif cfg.data.format == 'continuous':
        data_cls = PRMContinuosScoreDataset
    elif cfg.data.format == 'math-shepherd':
        data_cls = MathShepherdDataset
    elif cfg.data.format == 'future-sampling':
        data_cls = FutureSamplingDataset
    elif cfg.data.format == 'generative':
        data_cls = GenerativeInterleavedPRMDataset
    else:
        raise ValueError("Invalid data format: {}".format(cfg.data.format))
        
    dev_dataset = MathShepherdDataset(
        tokenizer=tokenizer,
        config=cfg.data,
        split='dev',
        )

    dev_questions = [d['question'] for d in dev_dataset.data]

    ## save tokenizer 
    ## load and construct training dataset
    if isinstance(cfg.data.data_dir, list):
        data_path = [os.path.join(p, 'train') for p in cfg.data.data_dir]
    else:
        data_path = [os.path.join(cfg.data.data_dir, 'train')]

    train_dataset = data_cls(
        data_path=data_path,
        tokenizer=tokenizer,
        config=cfg.data,
        split='train',
        )
    
    # iterate over datasets train_dataset for debugging
    for d in train_dataset:
        print(d)
    
    import ipdb; ipdb.set_trace()
    
    print("Filtering out dev questions from training data")
    old_train_size = len(train_dataset)
    ### keep only questions that are not in dev questions
    train_dataset.data = [d for d in train_dataset.data if d['question'] not in dev_questions]
    print("Filtered out {} dev questions from training data".format(old_train_size - len(train_dataset)))

    ## make sure no overlap between train and dev questions TODO

    print("Got {} training examples".format(len(train_dataset)))
    print("Got {} dev examples".format(len(dev_dataset)))

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    ### init wandb 
    wandb.init(project='mcts-prm', config=cfg)

    cfg.train.output_dir = os.path.join(cfg.train.output_dir, wandb.run.name)
    ### make sure output_dir exists
    os.makedirs(cfg.train.output_dir, exist_ok=True)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        print("Setting gradient accumulation steps to {}".format(cfg.train.gradient_accumulation_steps // world_size))
        cfg.train.gradient_accumulation_steps = cfg.train.gradient_accumulation_steps // world_size

    if cfg.model.lora.use: 
        # load the base model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name_or_path, 
            trust_remote_code=True,
            load_in_8bit=True,
            torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float16,
            output_attentions=False,
            device_map=device_map,
            )

        print("🤯 initializing LoRA model 🤯")
        ### prepare model for int8 training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.train.gradient_checkpointing)
        config = LoraConfig(
            r = cfg.model.lora.r,
            lora_alpha=cfg.model.lora.alpha,
            target_modules=list(cfg.model.lora.target_modules),
            task_type="SEQ_CLS",
            modules_to_save=list(cfg.model.lora.modules_to_save) if cfg.model.lora.get('modules_to_save', None) else None,
            bias="none",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    else: 
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.model_name_or_path,
            output_attentions=False,
            output_hidden_states=False,
            torch_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float16,
            device_map=device_map,
        )
        
    model.config.pad_token_id = model.config.eos_token_id

    if cfg.data.format == 'pairwise_stepwise':
        reward_config = RewardConfig(**cfg.train.to_dict(), 
                                    remove_unused_columns=False, 
                                    metric_for_best_model='f1', 
                                    greater_is_better=True,
                                    load_best_model_at_end=True)

        # load trainer
        trainer = RewardTrainer(
            args=reward_config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            data_collator=train_dataset.collate_fn,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )
    
    elif cfg.data.format in ['binary', 'math-shepherd', 'continuous', 'future-sampling']: # CE loss
        training_args = TrainingArguments(**cfg.train.to_dict(), remove_unused_columns=False, 
                                        metric_for_best_model='f1', 
                                        greater_is_better=True,
                                        load_best_model_at_end=True)

        trainer = PRMCustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            data_collator=train_dataset.collate_fn,
            compute_metrics=compute_metrics,
            pos_label_token_id=train_dataset.pos_step_id,
            neg_label_token_id=train_dataset.neg_step_id,
            label_format='continuous' if cfg.data.format == 'continuous' else 'binary'
        )

    elif cfg.data.format == 'cot':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=train_dataset.collate_fn,
            compute_metrics=compute_metrics,
        )

    else: 
        raise ValueError("Invalid data format: {}".format(cfg.data.format))

    ## pretty print training args
    print("Training args:")
    for arg, value in sorted(vars(training_args).items()):
        print(f"  {arg}: {value}")

    tr_eval_cb = TrainEvalCallback(trainer)
    trainer.add_callback(tr_eval_cb)
    trainer.train()
    ### save best model and tokenizer
    trainer.save_model(os.path.join(cfg.train.output_dir, 'best_model'))
    tokenizer.save_pretrained(os.path.join(cfg.train.output_dir, 'best_model'))
    
    # save config
    with open(os.path.join(cfg.train.output_dir, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f)

    if getattr(cfg.train, 'eval_when_finished', True):
        try:
            print("**************************************** EVALUATING BEST MODEL ****************************************")
            #### evaluate the model as best_of_n by calling the bash script 
            subprocess.run(
                ['bash', 'bscripts/paper/eval_prm_gsm8k.sh', os.path.join(cfg.train.output_dir, 'best_model')],
                env={**os.environ, 'PYTHONPATH': '.', 'CUDA_VISIBLE_DEVICES': '0,1'},
                    check=True
                )
            print("**************************************** EVALUATION COMPLETED ****************************************")
        except subprocess.CalledProcessError as e:
            print("An error occurred during evaluation:", e)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_prm.py <path_to_config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    cfg = Config.load_config(config_path)
    main(cfg)