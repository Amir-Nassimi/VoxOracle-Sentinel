import os
import librosa
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


class AudioFeatureExtractor:
    def __init__(self, audio_dir, save_numpy_dir, save_pickle_path, target_len, sample_rate=16000, n_mels=128):
        self.audio_dir = audio_dir
        self.save_numpy_dir = save_numpy_dir
        self.save_pickle_path = save_pickle_path
        self.target_len = target_len
        self.sample_rate = sample_rate
        self.n_mels = n_mels

    @staticmethod
    def standardize_audio_length(audio, target_length, sample_rate):
        target_samples = int(target_length * sample_rate)
        if len(audio) < target_samples:
            audio = librosa.util.fix_length(audio, size=target_samples)
        else:
            audio = audio[:target_samples]
        return audio

    def extract_mel_spectrogram(self, audio_input):
        if isinstance(audio_input, str):
            audio, sr = librosa.load(audio_input, sr=self.sample_rate, mono=True)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input
            sr = self.sample_rate
        else:
            raise TypeError("audio_input must be a file path (str) or an audio array (numpy.ndarray)")

        audio = self.standardize_audio_length(audio, self.target_len, sr)
        mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, hop_length=160, win_length=400, n_fft=512)
        spec_db_2 = librosa.amplitude_to_db(mel_spect) ** 2
        rgb_spec_db_2 = np.repeat(spec_db_2[..., np.newaxis], 3, -1)
        return rgb_spec_db_2

    def extract_features(self):
        x_list, y_list = [], []
        for subdir, dirs, files in os.walk(self.audio_dir):
            for file in tqdm(files):
                if file.endswith('.wav'):
                    audio_file = os.path.join(subdir, file)
                    audio_data_type = os.path.basename(subdir)
                    name = os.path.splitext(file)[0] + '.npy'

                    mel_spect = self.extract_mel_spectrogram(audio_file)
                    output_dir = os.path.join(self.save_numpy_dir, audio_data_type)
                    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                    np_path = os.path.join(output_dir, name)
                    np.save(np_path, mel_spect)
                    x_list.append(np_path)
                    y_list.append(audio_data_type)

        data_dict = {'x': x_list, 'y': y_list}
        data_frame = pd.DataFrame(data_dict)
        data_frame.to_pickle(f'{self.save_pickle_path}/data.pkl')


def main():
    parser = argparse.ArgumentParser(description="Extract features from audio files.")

    parser.add_argument('--audio_dir', type=str, required=True, help='Root directory path of audio files')
    parser.add_argument('--numpy_dir', type=str, required=True, help='Path to save numpy files')
    parser.add_argument('--pickle_path', type=str, required=True, help='Path to save pickle file')
    parser.add_argument('--target_len', type=float, required=True, help='The target length of audio file')

    args = parser.parse_args()

    extractor = AudioFeatureExtractor(args.audio_dir, args.numpy_dir, args.pickle_path, args.target_len)
    extractor.extract_features()


if __name__ == "__main__":
    main()
