import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.densenet import preprocess_input


class DataPreparation(Sequence):
    def __init__(self, csv_file, batch_size, dim, n_channels, n_classes, shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.data = pd.read_csv(csv_file)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data))
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.data['label'].tolist())
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_data = self.data.iloc[indexes]
        X, y = self.__data_generation(temp_data)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_data):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, row in enumerate(temp_data.itertuples()):
            spect = np.load(row.file_path)
            spect = np.stack((spect, spect, spect), axis=-1)  # Convert to 3 channels
            spect = preprocess_input(spect)  # Preprocess for DenseNet

            X[i,] = spect
            y[i] = self.label_encoder.transform([row.label])[0]

        return X, to_categorical(y, num_classes=self.n_classes)
