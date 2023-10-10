import pandas as pd
import argparse
import json
from transformers import pipeline
from datasets import Dataset

def normalize(text: str) -> str:
  text = text.lower()
  text = text.replace("_", " ")
  return text

def get_pipeline(checkpoint_path: str):
    classifier = pipeline("ner", checkpoint_path, 
                         device=0, aggregation_strategy='simple')
    return classifier

def get_dataset(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].apply(normalize)
    return Dataset.from_pandas(df)

def inference(checkpoint_path: str, dataset_path: str):
   classifier = get_pipeline(checkpoint_path)
   dataset = get_dataset(dataset_path)
   return classifier(dataset['text'])

def count_num_type_entities(result: list):
    entities_list = []
    count = 0
    for entity in result:
        if entity['entity_group'] not in entities_list:
            count += 1
            entities_list.append(entity['entity_group'])
    return count

def get_result_by_num_type_entites(dataset_path: str, result: list):
    result_by_num_entities = {"index":[], "text": [], "intent": [], "num_entities":[]}
    df = pd.read_csv(dataset_path)
    df['text'] = df['text'].apply(normalize)
    for i in range(len(result)):
        result_by_num_entities["index"].append(i)
        result_by_num_entities["text"].append(df["text"].values[i])
        result_by_num_entities["intent"].append(df["label"].values[i])
        result_by_num_entities["num_entities"].append(count_num_type_entities(result[i]))
    
    return pd.DataFrame(result_by_num_entities)

def create_pseudo_label(df:pd.DataFrame, result:list):
    result_object = {"rasa_nlu_data":{"common_examples":[]}}
    original_df = df
    df = df.drop(df[df['intent'] == 'chitchat_ask_math'].index)
    df = df.sort_values('num_entities', ascending=False)
    df = df.iloc[:7000]
    for index in df['index']:
        entities_list = []
        for entity in result[index]:
            if entity['entity_group'] == "ROLE":
                entity['entity_group'] = "POSITION"
            entities_list.append({
                "end":entity['end'],
                "entity": entity['entity_group'],
                "start": entity['start'],
                "value": entity["word"],
            })
        added_object = {}
        added_object['entities'] = entities_list
        added_object['intent'] = original_df.iloc[index]["intent"]
        added_object['text'] = original_df.iloc[index]["text"]
        added_object['source'] = "train"
        result_object['rasa_nlu_data']['common_examples'].append(added_object)
    return result_object

def write_json_file(result_object: dict):
    new_json_object = json.dumps(result_object,ensure_ascii=False).encode('utf8')
    with open("sample.json", "w", encoding='utf8') as outfile:
        outfile.write(new_json_object.decode())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--data-directory", type=str, required=True)
    args = parser.parse_args()

    result = inference(checkpoint_path=args.checkpoint_path, dataset_path=args.data_directory)
    result_by_num_entities = get_result_by_num_type_entites(dataset_path=args.data_directory, result=result)

    result_object = create_pseudo_label(result_by_num_entities, result)
    write_json_file(result_object=result_object)