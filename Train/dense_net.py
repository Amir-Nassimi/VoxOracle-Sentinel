import os
import math
from datetime import datetime
from singleton_decorator import singleton

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout


@singleton
class ModelBuilder:
    def __init__(self, input_shape):
        self.base_model = DenseNet121(weights='imagenet', include_top=False, pooling='avg', input_shape=input_shape)

    def build_model(self, no_class):
        x = BatchNormalization()(self.base_model.output)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        preds = Dense(no_class, activation='softmax')(x)

        model = Model(inputs=self.base_model.input, outputs=preds)
        return model


class TrainingManager:
    def __init__(self, model, checkpoint_dir, logdir, train_data, valid_data):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.logdir = logdir
        self.train_data = train_data
        self.valid_data = valid_data

    def train(self, batch_size, epochs):
        x_train, y_train = self.train_data
        x_valid, y_valid = self.valid_data

        n_batches = math.ceil(len(x_train) / batch_size)
        checkpoint_filepath = f'{self.checkpoint_dir}/ckpt'

        model_checkpoint_callback = ModelCheckpoint(
            checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            save_freq=50 * n_batches)

        tensorboard_callback = TensorBoard(log_dir=os.path.join(self.logdir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid),
                       callbacks=[model_checkpoint_callback, tensorboard_callback])
