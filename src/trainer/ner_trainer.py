import pandas as pd
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
)
from datasets import Dataset
from src.data.text_processing import *
from src.data.data_processing import *
from src.evaluate import compute_metrics


class NerTrainer:
    def __init__(self, model_name: str, dataset_path: str):
        self.model_name = model_name
        self.df = pd.read_csv(dataset_path)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.df["text_x"] = self.df["text_x"].apply(text_normalize)
        self.dataset = Dataset.from_pandas(self.df)

    def _init_dataset(self):
        tokenized_dataset = self.dataset.map(tokenize_and_align_label, batched=True)
        return tokenized_dataset

    def _load_trainer(self):
        tokenized_dataset = self._init_dataset()
        training_args = TrainingArguments(
            f"timi-ner",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            num_train_epochs=5,
            warmup_steps=500,
            logging_dir="./logs",
            logging_steps=100,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            load_best_model_at_end=True,
        )
        id2label = {i: label for i, label in enumerate(NER_TAGS)}
        label2id = {v: k for k, v in id2label.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, id2label=id2label, label2id=label2id
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        return trainer

    def train(self):
        trainer = self._load_trainer()
        trainer.train()


if __name__ == "__main__":
    trainer = NerTrainer(
        model_name=model_name, dataset_path="./dataset/NER_dataset.csv"
    )
    trainer.train()
