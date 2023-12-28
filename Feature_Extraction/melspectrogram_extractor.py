import os
import librosa
import pathlib
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def standardize_audio_length(audio, target_length, sample_rate):
    # Calculate the target number of samples
    target_samples = int(target_length * sample_rate)

    # Pad or truncate the audio signal
    if len(audio) < target_samples:
        # Pad audio if shorter than target length
        audio = librosa.util.fix_length(audio, size=target_samples)
    else:
        # Truncate audio if longer than target length
        audio = audio[:target_samples]

    return audio


def extract_mel_spectrogram(audio_input, target_len, sample_rate=16000, n_mels=128):
    if isinstance(audio_input, str):
        audio, sr = librosa.load(audio_input, sr=sample_rate, mono=True)
    elif isinstance(audio_input, np.ndarray):
        audio = audio_input
        sr = sample_rate
    else:
        raise TypeError("audio_input must be a file path (str) or an audio array (numpy.ndarray)")

    # Normalize if necessary
    if np.issubdtype(audio.dtype, np.floating) and np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    elif not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32) / np.max(np.abs(audio))

    print(f'Audio length: {len(audio) / sample_rate}')
    audio = standardize_audio_length(audio, target_len, sr)
    print(f'Final Audio length: {len(audio) / sample_rate}\n')

    # Extract Mel spectrogram
    mel_spect = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=160, win_length=400, n_fft=512)

    spec_db_2 = librosa.amplitude_to_db(mel_spect) ** 2
    rgb_spec_db_2 = np.repeat(spec_db_2[..., np.newaxis], 3, -1)

    return rgb_spec_db_2


def extract_features(audio_dir, save_numpy_dir, save_pickle_path, target_len, sample_rate=16000, n_mels=128):
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

                mel_spect = extract_mel_spectrogram(audio_file, target_len, sample_rate, n_mels)
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

    parser.add_argument('--numpy_dir', type=str, required=True, help='Path to save numpy files')
    parser.add_argument('--pickle_path', type=str, required=True, help='Path to save pickle file')
    parser.add_argument('--audio_dir', type=str, required=True, help='Root directory path of audio files')
    parser.add_argument('--target_len', type=float, required=True, help='The target length of audio file')

    args = parser.parse_args()

    extract_features(args.audio_dir, args.numpy_dir, args.pickle_path, args.target_len)


if __name__ == "__main__":
    main()

