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
        _, y = self.train_data[0]
        #n_batches = math.ceil(len(y) / batch_size)
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
                       callbacks=[model_checkpoint_callback, tensorboard_callback])
