from utils import get_model_path, load_classifier, apply_classifier_prompt
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
import torch, random, json, os
from datasets import Dataset
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary', zero_division=0)
    recall = recall_score(labels, predictions, average='binary')
    alpha = 2
    w_p = 1 if precision >= 0.70 else np.exp(-5 * (0.70 - precision))
    w_r = 1 if recall >= 0.5 else np.exp(-7 * (0.5 - recall))
    f1 = (1 + alpha) * (w_p * precision) * (w_r * recall) / (w_p * precision + alpha * w_r * recall)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

class WeightedCELossTrainer(Trainer):
    def __init__(self, model, args, neg_rate=1.0, pos_rate=1.0, **kwargs):
        super().__init__(model, args, **kwargs)
        if neg_rate == 0 or pos_rate == 0:
            self.neg_weights = 1
            self.pos_weights = 1
        else:
            self.neg_weights = 1 / neg_rate
            self.pos_weights = 1 / pos_rate
            print(f"Negative weight: {self.neg_weights}, Positive weight: {self.pos_weights}")
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        assert(num_items_in_batch == 1 or num_items_in_batch is None)
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([self.neg_weights, self.pos_weights], device=model.device, dtype=logits.dtype)
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
def process_data_without_split(datas, shuffle=True):
    pos_data, neg_data = [], []
    for data in datas:
        label = data["label"]
        assert label in [0, 1]
        text = apply_classifier_prompt(data["data"])
        if label:
            pos_data.append({"text": text, "label": label})
        else:
            neg_data.append({"text": text, "label": label})
    neg_rate = len(neg_data) / len(datas) if len(datas) > 0 else None
    pos_rate = len(pos_data) / len(datas) if len(datas) > 0 else None
    print(f"Negative data size: {len(neg_data)}, Positive data size: {len(pos_data)}")
    # shuffle data
    merged_datas = neg_data + pos_data
    if shuffle:
        random.shuffle(merged_datas)
    print(f"Data size: {len(datas)}")
    return merged_datas, neg_rate, pos_rate

def unify_checkpoint_dir(peft_model_train_dir, peft_model_eval_dir):
    # find all checkpoint-<num> dir in peft_model_dir and reserve the last one
    checkpoint_list = [f for f in os.listdir(peft_model_train_dir) if f.startswith("checkpoint-")]
    # sort by num suffix of integer format
    checkpoint_list.sort(key=lambda x: int(x.split("-")[1]))
    # reserve the biggest one and delete others
    for checkpoint in checkpoint_list[:-1]:
        checkpoint_path = os.path.join(peft_model_train_dir, checkpoint)
        os.system(f"rm -rf {checkpoint_path}")
    # rename the last checkpoint dir to checkpoint-0
    last_checkpoint_path = os.path.join(peft_model_train_dir, checkpoint_list[-1])
    os.rename(last_checkpoint_path, peft_model_eval_dir)

def refine_model(output_dir, tokenizer, classifier, model_name, neg_rate, pos_rate, train_dataset, eval_dataset, epochs=30):
    
    classifier_train_dir = get_model_path(model_name, base_dir=output_dir, has_suffix=False)
    classifier_eval_dir = get_model_path(model_name, base_dir=output_dir, has_suffix=True)
    tokenized_dataset = {}
    tokenized_dataset["train"] = train_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    tokenized_dataset["eval"] = eval_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True), batched=True)
    print(f"Train size: {len(tokenized_dataset['train'])}, Eval size: {len(tokenized_dataset['eval'])}")
    
    training_args = TrainingArguments(
        output_dir=classifier_train_dir,
        warmup_ratio= 0.1,
        max_grad_norm= 0.3,
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        weight_decay=0.001,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        disable_tqdm=True,
        evaluation_strategy="epoch",
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=1,
        warmup_steps=500,
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = WeightedCELossTrainer(
        model=classifier,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=training_args,
        neg_rate=neg_rate,
        pos_rate=pos_rate,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )
    
    trainer.train()
    
    unify_checkpoint_dir(classifier_train_dir, classifier_eval_dir)
    
    
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="deepseek-7b-chat")
    parser.add_argument("--base_model_dir", type=str, default="base_model")
    parser.add_argument("--output_dir", type=str, default="ensemble")
    args = parser.parse_args()
    with open("../data/train.json", "r") as f:
        train_data = json.load(f)
    with open("../data/val.json", "r") as f:
        val_data = json.load(f)
    train_data, neg_rate, pos_rate = process_data_without_split(train_data, shuffle=True)
    val_data, _, _ = process_data_without_split(val_data, shuffle=True)
    train_dataset = Dataset.from_dict({key: [data[key] for data in train_data] for key in train_data[0]})
    val_dataset = Dataset.from_dict({key: [data[key] for data in val_data] for key in val_data[0]})
    model_name = args.model_name
    output_dir = args.output_dir
    tokenizer, classifier = load_classifier(args.base_model_dir, model_name, state="train")
    refine_model(output_dir, tokenizer, classifier, model_name, neg_rate, pos_rate, train_dataset, val_dataset)