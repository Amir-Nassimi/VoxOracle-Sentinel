import os
import sys
import numpy as np
from pathlib import Path
from collections import deque
from singleton_decorator import singleton
from tensorflow.keras.applications.densenet import preprocess_input

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Feature_Extraction.melspectogram_extractor import AudioFeatureExtractor


@singleton
class FrameASR:
    def __init__(self, model, frame_len, target_len, frame_overlap, label_source, pr_acc, sample_rate=16000):
        self.model = model
        self.pr_acc = pr_acc / 100
        self.frame_len = frame_len
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.label_source = label_source
        self.frame_overlap = frame_overlap

        self.feature_extractor = AudioFeatureExtractor('', '', '', target_len, sample_rate)

        self.n_frame_len = int(frame_len * self.sample_rate)
        self.n_frame_overlap = int(frame_overlap * self.sample_rate)

        self.buffer_len = (2*self.n_frame_overlap) + self.n_frame_len
        self.buffer = deque(np.zeros(self.buffer_len), maxlen=self.buffer_len)
        self.reset()

    def reset(self):
        self.buffer.extend(np.zeros(self.buffer_len))

    def transcribe(self, frame):
        if len(frame) == 0:
            frame = np.zeros(self.n_frame_len, dtype=np.float32)
        # elif len(frame) < self.n_frame_len:
        #     frame = np.pad(frame, (0, self.n_frame_len - len(frame)), 'constant')

        # Update buffer with new frame
        # self.buffer = np.roll(self.buffer, -self.n_frame_len)
        # self.buffer[-self.n_frame_len:] = frame
        self.buffer.extend(frame)

        spect = self.feature_extractor.extract_mel_spectrogram(np.array(self.buffer))
        spect = preprocess_input(spect)

        try:
            result = self.model.predict(np.expand_dims(spect, axis=0))[0]
            print(result)
        except Exception as error:
            raise ValueError(f'Error: {error}')
        return self.decode_pred(result)

    def decode_pred(self, result):
        if np.max(result) < self.pr_acc:
            return 'unknown'
        else:
            label_num = np.argmax(result)
            return self.label_source.get(label_num, "unknown")
