import os
import sys
import argparse
from pathlib import Path
from dense_net import ModelBuilder, TrainingManager, EvaluationManager

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from data_proc import DataPreparation


class HowToRDTrainer:
    def __init__(self, **kwargs):
        self.setup_environment()

        self.epoch = kwargs.get('epoch')
        self.batch = kwargs.get('b_size')
        self.log_dir = kwargs.get('logdir')
        self.classes = kwargs.get('classes')
        self.checkpoint_dir = kwargs.get('checkpoint_dir')
        self.in_shape = tuple(map(int, kwargs.get('in_shape').split(',')))

        self.train_gen = DataPreparation(kwargs.get('train_csv'),
                                         batch_size=self.batch,
                                         dim=self.in_shape,
                                         n_channels=3,
                                         n_classes=self.classes,
                                         shuffle=True)

        self.valid_gen = DataPreparation(kwargs.get('valid_csv'),
                                         batch_size=self.batch,
                                         dim=self.in_shape,
                                         n_channels=3,
                                         n_classes=self.classes,
                                         shuffle=True)

        self.test_gen = DataPreparation(kwargs.get('test_csv'),
                                        batch_size=self.batch,
                                        dim=self.in_shape,
                                        n_channels=3,
                                        n_classes=self.classes,
                                        shuffle=False)

        label_to_index_mapping = {index: label for index, label in enumerate(self.valid_gen.label_encoder.classes_)}
        print(f'Labels:\n{label_to_index_mapping}')

        self.model_builder = ModelBuilder(input_shape=self.in_shape+(3,))

    def setup_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    def run(self):
        model = self.model_builder.build_model(self.classes)
        training_manager = TrainingManager(model, self.checkpoint_dir, self.log_dir,
                                           self.train_gen, self.valid_gen, self.test_gen)
        training_manager.train(batch_size=self.batch, epochs=self.epoch)

        test_manager = EvaluationManager(training_manager.model, self.test_gen)
        test_manager.evaluate()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type=str, required=True, help='path to save logs')
    parser.add_argument('--epoch', type=int, required=False, default=100, help='Epoch size')
    parser.add_argument('--b_size', type=int, required=False, default=16, help='Batch size')
    parser.add_argument('--classes', type=int, required=True, help='No. of classes to train')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='path to save checkpoints.')
    parser.add_argument('--train_csv', type=str, required=True, help='path to .csv file for train dataset')
    parser.add_argument('--valid_csv', type=str, required=True, help='path to .csv file for validation dataset')
    parser.add_argument('--test_csv', type=str, required=True, help='path to .csv file for test dataset')
    parser.add_argument('--in_shape', type=str, default='128,191', required=False, help='The input shape of the model')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    trainer = HowToRDTrainer(**vars(args))
    trainer.run()
