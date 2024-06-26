import sys, os
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Train.dense_net import ModelBuilder
from Train.data_proc import DataPreparation
from Train.managers import TrainingManager, EvaluationManager


class HowToRDTrainer:
    def __init__(self, **kwargs):
        self.setup_environment()

        self.epoch = kwargs.get('epoch')
        self.batch = kwargs.get('b_size')
        self.log_dir = kwargs.get('logdir')
        self.classes = kwargs.get('classes')
        self.softmax_type = kwargs.get('softmax_type')
        self.attention_type = kwargs.get('attention_type')
        self.checkpoint_dir = kwargs.get('checkpoint_dir')
        self.initial_sparsity_rate = kwargs.get('initial_sparsity_rate')
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

    @staticmethod
    def setup_environment():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    def run(self):
        model = self.model_builder.build_model(self.classes,self.attention_type, self.softmax_type,
                                               self.initial_sparsity_rate)
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
    parser.add_argument('--in_shape', type=str, default='128,211', required=False, help='The input shape of the model')
    parser.add_argument('--initial_sparsity_rate', type=float, required=False, default=0.5,
                        help='The Initial Sparsity Rate of R_Softmax; default value: 0.5')
    parser.add_argument('--softmax_type', type=str, required=False, default='normal',
                        help='The Softmax Strategy to use; Available Strategies: "normal" (Softmax), "r_softmax" (R_Softmax)')
    parser.add_argument('--attention_type', type=str, required=False, default='no_attention',
                        help='Whether to utilize an Attention Strategy or not; "normal" (Normal Attention Layer), "gaap" (GAAP Attention Layer), default: not to utilize any')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    trainer = HowToRDTrainer(**vars(args))
    trainer.run()
