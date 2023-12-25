import numpy as np
import pandas as pd
from singleton_decorator import singleton
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


@singleton
class DataPreparation:
    def __init__(self, train_csv, valid_csv):
        self.train_csv = train_csv
        self.valid_csv = valid_csv

    def load_data(self):
        train_data = pd.read_csv(self.train_csv)
        valid_data = pd.read_csv(self.valid_csv)

        x_train, y_train = self.process_data(train_data)
        x_valid, y_valid = self.process_data(valid_data, train=False)

        return x_train, y_train, x_valid, y_valid

    def process_data(self, data, train=True):
        audio_paths = np.array(data['file_path'].tolist())  # x
        classes = np.array(data['label'].tolist())  # y

        audio_list = [np.load(audio) for audio in audio_paths]
        audio_array = np.array(audio_list)

        if train:
            self.labelencoder = LabelEncoder()
            classes = to_categorical(self.labelencoder.fit_transform(classes))
        else:
            classes = to_categorical(self.labelencoder.transform(classes))

        return audio_array, classes