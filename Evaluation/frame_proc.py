import os
import sys
import numpy as np
from pathlib import Path
from singleton_decorator import singleton
from tensorflow.keras.applications.densenet import preprocess_input

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Feature_Extraction.melspectrogram_extractor import extract_mel_spectrogram


@singleton
class FrameASR:
    def __init__(self, model, frame_len, target_len, frame_overlap, label_source, sample_rate=16000):
        self.model = model
        self.frame_len = frame_len
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.label_source = label_source
        self.frame_overlap = frame_overlap

        self.n_frame_len = int(frame_len * self.sample_rate)
        self.n_frame_overlap = int(frame_overlap * self.sample_rate)

        buffer_len = (2*self.n_frame_overlap) + self.n_frame_len
        self.buffer = np.zeros(buffer_len, dtype=np.float32)
        self.reset()

    def reset(self):
        self.buffer.fill(0.0)  # Reset buffer

    def transcribe(self, frame):
        if len(frame) == 0:
            frame = np.zeros(self.n_frame_len, dtype=np.float32)
        elif len(frame) < self.n_frame_len:
            frame = np.pad(frame, (0, self.n_frame_len - len(frame)), 'constant')

        # Update buffer with new frame
        self.buffer = np.roll(self.buffer, -self.n_frame_len)
        self.buffer[-self.n_frame_len:] = frame

        spect = extract_mel_spectrogram(self.buffer, self.target_len, self.sample_rate)
        spect = preprocess_input(spect)

        result = self.model.predict(np.expand_dims(spect, axis=0))[0]
        return self.decode_pred(result)

    def decode_pred(self, result):
        label_num = np.argmax(result)
        return self.label_source.get(label_num, "unknown")