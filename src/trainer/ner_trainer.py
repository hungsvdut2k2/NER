import torch
from transformers import (DataCollatorForTokenClassification, AutoTokenizer, 
                          TrainingArguments, Trainer, AutoModelForTokenClassification)
from src.trainer.base_trainer import BaseTrainer
from src.metrics import compute_metrics
from src.constant import id2label, label2id

class NerTrainer(BaseTrainer):
    def __init__(self, model_name: str, data_directory: str) -> None:
        super().__init__(model_name=model_name, data_directory=data_directory)
    
    def _load(self, model_directory: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_directory, id2label=id2label, label2id=label2id)
        return tokenizer, model

    def _train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model =  AutoModelForTokenClassification.from_pretrained(self.model_name, id2label=id2label, 
                                                                 label2id=label2id)
        dataset = self._load_dataset(trainer_type="ner")
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        training_args = TrainingArguments(
            output_dir="timi-ner",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["valid"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()