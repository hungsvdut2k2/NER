import argparse
from src.trainer.trainer_factory import TrainerFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="vinai/phobert-base-v2")
    parser.add_argument("--data-directory", type=str, required=True)
    parser.add_argument("--trainer-type", type=str, default="ner")
    args = parser.parse_args()
    trainer_factory = TrainerFactory(model_name=args.model_name,
                                     data_directory=args.data_directory)
    trainer = trainer_factory.get_trainer(format=args.trainer_type)
    trainer.train()    