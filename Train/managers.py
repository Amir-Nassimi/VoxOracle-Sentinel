import os
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from train_handlers import CustomCheckpoint, TestSetEvaluationCallback, StopAtAccuracy


class TrainingManager:
    def __init__(self, model, checkpoint_dir, logdir, train_data, valid_data, test_data):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data

        self.checkpoints(logdir, checkpoint_dir, test_data)

    def checkpoints(self,logdir, checkpoint_dir, test_data):
        checkpoint_filepath = f'{checkpoint_dir}/ckpt.hdf5'

        self.model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,  # Save only the best model
            save_weights_only=True,  # Change to True if you want to save only weights
            verbose=1,
            save_freq='epoch')

        self.evaluation_callback = TestSetEvaluationCallback(test_data, f'{logdir}/Test', checkpoint_dir)
        self.custom_checkpoint_callback = CustomCheckpoint(filepath=checkpoint_dir, monitor='val_accuracy')
        self.tensorboard_callback = TensorBoard(log_dir=os.path.join(f'{logdir}/Train',
                                                                datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
        self.stop_callback = StopAtAccuracy()

    def train(self, batch_size, epochs):
        _, y = self.train_data[0]

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.train_data, epochs=epochs, batch_size=batch_size, validation_data=self.valid_data,
                       callbacks=[self.model_checkpoint_callback, self.tensorboard_callback, self.evaluation_callback,
                                  self.custom_checkpoint_callback, self.stop_callback])


class EvaluationManager:
    def __init__(self, model, test_set):
        self.model = model
        print('Model loaded successfully!')
        self.test_data = test_set

    def evaluate(self):
        test_loss, test_accuracy = self.model.evaluate(self.test_data, verbose=1)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
