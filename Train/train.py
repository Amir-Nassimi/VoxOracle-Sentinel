import os
import argparse
from data_proc import DataPreparation
from dense_net import ModelBuilder, TrainingManager


class HowToRDTrainer:
    def __init__(self, **kwargs):
        self.setup_environment()

        self.epoch = kwargs.get('epoch')
        self.batch = kwargs.get('b_size')
        self.log_dir = kwargs.get('logdir')
        self.classes = kwargs.get('classes')
        self.checkpoint_dir = kwargs.get('checkpoint_dir')
        self.data_preparation = DataPreparation(kwargs.get('train_csv'), kwargs.get('valid_csv'))
        self.model_builder = ModelBuilder(input_shape=(100, 301, 3))


    def setup_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    def run(self):
        x_train, y_train, x_valid, y_valid = self.data_preparation.load_data()
        model = self.model_builder.build_model(self.classes)
        training_manager = TrainingManager(model, self.checkpoint_dir, self.log_dir,
                                           (x_train, y_train), (x_valid, y_valid))
        training_manager.train(batch_size=self.batch, epochs=self.epoch)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, required=True, help='path to save logs')
    parser.add_argument('--epoch', type=int, required=False, default=100, help='Epoch size')
    parser.add_argument('--b_size', type=int, required=False, default=16, help='Batch size')
    parser.add_argument('--classes', type=int, required=True, help='No. of classes to train')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='path to save checkpoints.')
    parser.add_argument('--train_csv', type=str, required=True, help='path to .csv file for train dataset')
    parser.add_argument('--valid_csv', type=str, required=True, help='path to .csv file for validation dataset')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    trainer = HowToRDTrainer(**vars(args))
    trainer.run()
