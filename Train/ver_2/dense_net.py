import os
from datetime import datetime
from singleton_decorator import singleton

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.applications import DenseNet121
from tensorflow.summary import create_file_writer, scalar
from tensorflow.keras.backend import tanh, dot, softmax
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.W = None
        self.b = None

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(1, input_shape[-1]),
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tanh(dot(x, self.W) + self.b)
        a = softmax(e, axis=1)
        output = x * a
        return output

    def compute_output_shape(self, input_shape):
        return input_shape


@singleton
class ModelBuilder:
    def __init__(self, input_shape):
        self.base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

    def build_model(self, no_class):
        x = self.base_model.output
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        x = AttentionLayer()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        preds = Dense(no_class, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=preds)
        return model


class TestSetEvaluationCallback(Callback):
    def __init__(self, test_data, log_dir, verbose=0, epoch_check=5):
        super(TestSetEvaluationCallback, self).__init__()
        self.verbose = verbose
        self.test_data = test_data
        self.epoch_check = epoch_check
        self.writer = create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.epoch_check == 0:
            test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=self.verbose)
            print(f"Testing:\n\tEpoch {epoch + 1}: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            with self.writer.as_default():
                scalar('test_loss', test_loss, step=epoch)
                scalar('test_accuracy', test_accuracy, step=epoch)
                self.writer.flush()


class TrainingManager:
    def __init__(self, model, checkpoint_dir, logdir, train_data, valid_data, test_data):
        self.model = model
        self.logdir = f'{logdir}/Train'
        self.train_data = train_data
        self.valid_data = valid_data
        self.checkpoint_dir = checkpoint_dir
        self.evaluation_call_back = TestSetEvaluationCallback(test_data, f'{logdir}/Test')

    def train(self, batch_size, epochs):
        _, y = self.train_data[0]
        checkpoint_filepath = f'{self.checkpoint_dir}/ckpt.h5'

        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,  # Save only the best model
            save_weights_only=True,  # Change to True if you want to save only weights
            verbose=1,
            save_freq='epoch')

        tensorboard_callback = TensorBoard(log_dir=os.path.join(self.logdir,
                                                                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, epochs=epochs, batch_size=batch_size, validation_data=self.valid_data,
                       callbacks=[model_checkpoint_callback, tensorboard_callback, self.evaluation_call_back])


class EvaluationManager:
    def __init__(self, model, test_set):
        self.model = model
        print('Model loaded successfully!')
        self.test_data = test_set

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=1)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")