import os
import librosa
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_mel_spectrogram(audio_file, sample_rate=16000, n_mels=128):
    """
    Extract a Mel spectrogram from an audio file.
    """
    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    # Extract Mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    return mel_spect


def extract_features(audio_dir, save_numpy_dir, save_pickle_path, sample_rate=16000, n_mels=128):
    '''
    Extracting and storing the Mel spectrogram of each audio file.
    '''
    x_list, y_list = [], []
    for subdir, dirs, files in os.walk(audio_dir):
        for file in tqdm(files):
            if file.endswith('.wav'):
                audio_file = os.path.join(subdir, file)
                audio_data_type = os.path.basename(subdir)
                name = os.path.splitext(file)[0] + '.npy'

                mel_spect = extract_mel_spectrogram(audio_file, sample_rate, n_mels)
                output_dir = os.path.join(save_numpy_dir, audio_data_type)
                if not os.path.exists(output_dir):
                    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
                np_path = os.path.join(output_dir, name)
                np.save(np_path, mel_spect)
                x_list.append(np_path)
                y_list.append(audio_data_type)

    data_dict = {'x': x_list, 'y': y_list}
    data_frame = pd.DataFrame(data_dict)
    data_frame.to_pickle(f'{save_pickle_path}/data.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help='Root directory path of audio files')
    parser.add_argument('--numpy_dir', type=str, required=True, help='Path to save numpy files')
    parser.add_argument('--pickle_path', type=str, required=True, help='Path to save pickle file')
    args = parser.parse_args()

    extract_features(args.audio_dir, args.numpy_dir, args.pickle_path)


if __name__ == "__main__":
    main()

