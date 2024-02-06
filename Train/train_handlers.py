import os
import numpy as np
from tensorflow.summary import create_file_writer, scalar
from tensorflow.keras.callbacks import Callback, ModelCheckpoint


class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_accuracy'):
        super().__init__(filepath)
        self.filepath = filepath
        self.monitor = monitor
        self.best = -np.Inf
        self.filepath_template = '{pth}/{epoch}-{train_acc:.2f}_{val_acc:.2f}.hdf5'

    def on_epoch_end(self, epoch, logs=None):
        train_acc = logs['accuracy']
        val_acc = logs['val_accuracy']

        filepath = self.filepath_template.format(pth=self.filepath,
                                                 epoch=epoch,
                                                 train_acc=train_acc,
                                                 val_acc=val_acc)
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            pass
        elif np.greater(current, self.best):
            self.best = current
            self._save_model(epoch, filepath, logs)

    def _save_model(self, epoch, filepath, logs):
        self.model.save_weights(filepath, overwrite=False)


class TestSetEvaluationCallback(Callback):
    def __init__(self, test_data, log_dir, checkpoint_dir, verbose=0, epoch_check=1):
        super(TestSetEvaluationCallback, self).__init__()
        self.verbose = verbose
        self.test_data = test_data
        self.epoch_check = epoch_check
        self.writer = create_file_writer(log_dir)
        self.log_file = os.path.join(checkpoint_dir, "test_metrics.txt")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_check == 0:
            test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=self.verbose)
            print(f"Testing:\n\tEpoch {epoch + 1}: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            with self.writer.as_default():
                scalar('test_loss', test_loss, step=epoch)
                scalar('test_accuracy', test_accuracy, step=epoch)
                self.writer.flush()

            with open(self.log_file, "a") as f:
                f.write(f"Epoch {epoch + 1} - Loss: {test_loss} - Acc: {test_accuracy}\n")


class StopAtAccuracy(Callback):
    def __init__(self):
        super(StopAtAccuracy, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs['val_accuracy']
        if val_acc == 1.0:
            print(f"\nReached 100% validation accuracy at epoch {epoch}")
            self.model.stop_training = True
