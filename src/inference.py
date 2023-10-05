from transformers import pipeline
from seqeval.metrics import classification_report
from src.data.data_factory import DataFactory
from src.trainer.trainer_factory import TrainerFactory

def inference(trainer_type: str, model_name: str, 
              model_directory: str, data_directory: str, 
              type_dataset: str):
    data_factory = DataFactory(model_name=model_name)
    dataset_type  = data_factory.get_data(format=trainer_type)
    dataset  = dataset_type.load(data_directory=data_directory)[type_dataset]

    trainer_factory = TrainerFactory(model_name=model_name, data_directory=data_directory)
    trainer = trainer_factory.get_trainer(format=trainer_type)

    _, model = trainer.load(model_directory=model_directory)

    classifier =  pipeline(trainer_type, model=model)
    result = classifier.run(dataset['text'])
    print(result)

if __name__ == "__main__":
    inference(trainer_type="ner", model_name="vinai/phobert-base-v2", 
              model_directory="",data_directory="/home/bakerdn/NER/dataset/v0",
              type_dataset="test")