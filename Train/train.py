import os
import argparse
from data_proc import DataPreparation
from dense_net import ModelBuilder, TrainingManager


class HowToRDTrainer:
    def __init__(self, args):
        self.args = args
        self.setup_environment()
        self.data_preparation = DataPreparation(args.train_csv, args.valid_csv)
        self.model_builder = ModelBuilder(input_shape=(100, 301, 3))
        self.training_manager = None

    def setup_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    def run(self):
        x_train, y_train, x_valid, y_valid = self.data_preparation.load_data()
        model = self.model_builder.build_model(self.args.classes)
        self.training_manager = TrainingManager(model, self.args.checkpoint_dir, self.args.logdir, (x_train, y_train), (x_valid, y_valid))
        self.training_manager.train(batch_size=16, epochs=150)


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logdir', type=str, required=True, help='path to save logs')
    parser.add_argument('--classes', type=int, required=True, help='No. of classes to train')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='path to save checkpoints.')
    parser.add_argument('--train_csv', type=str, required=True, help='path to .csv file for train dataset')
    parser.add_argument('--valid_csv', type=str, required=True, help='path to .csv file for validation dataset')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    trainer = HowToRDTrainer(args)
    trainer.run()
