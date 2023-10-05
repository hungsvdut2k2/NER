import os
from datasets import Dataset, DatasetDict
from src.data.base_data import BaseData
from src.constant import label2id

class NerData(BaseData):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name=model_name)
    
    def _load(self, data_directory: str):
        dataset = self._get_dataset_dict(data_directory)
        final_dataset = dataset.map(self._tokenize_and_align_labels, batched=True)
        return final_dataset
    
    def _get_dataset_dict(self, data_directory: str) -> Dataset:
        print(data_directory)
        return DatasetDict.from_csv({
            'train': os.path.join(data_directory, "train_ner.csv"),
            'valid': os.path.join(data_directory, "valid_ner.csv"),
            'test': os.path.join(data_directory,"test_ner.csv")
        })
    
    def _tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["text"], truncation=True)
        labels = []
        for i, label in enumerate(examples[f"tags"]):
            label = label.split()
            word_ids = tokenized_inputs.word_ids(batch_index=i) 
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs