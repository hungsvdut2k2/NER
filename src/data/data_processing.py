from src.constants import NER_TAGS, model_name
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(text):
    return tokenizer(text, truncation=True)


def mapping_label(labels):
    result = []
    for label in labels:
        temp_label = []
        for tag in label.split():
            temp_label.append(NER_TAGS[tag])
        result.append(temp_label)
    return result


def tokenize_and_align_label(examples):
    tokenized_inputs = tokenize(examples["text_x"])
    labels = mapping_label(examples["tags"])
    tokenized_inputs["labels"] = labels
    return tokenized_inputs
