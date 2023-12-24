import os
import random
import librosa
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
import soundfile as sf


def get_audio_files(directory):
    # For directories without a specific pattern
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]


def get_audio_files_recursive(directory):
    return glob(os.path.join(directory, '**', '*.wav'), recursive=True)


def process_audio(file1, file2, output_dir, sample_rate):
    # Load both audio files with the specified sample rate
    audio1, sr1 = librosa.load(file1, sr=sample_rate)
    audio2, _ = librosa.load(file2, sr=sample_rate)

    # Adjust the length of audio1 to match audio2
    len1, len2 = len(audio1), len(audio2)
    if len2 <= len1:
        # Randomly cut audio1 to the length of audio2
        start = random.randint(0, len1 - len2)
        audio1 = audio1[start:start + len2]
    else:
        # Skip this audio1 as it's shorter than audio2
        return False

    # Add the audio files
    mixed_audio = audio1 + audio2

    # Ensure the audio does not exceed -1 to 1 range
    mixed_audio = np.clip(mixed_audio, -1, 1)

    # Save the mixed audio
    output_filename = os.path.basename(file1).replace('.wav', '_mixed.wav')
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, mixed_audio, sr1)

    return True


def mix_audios(folder1, folder2, output_dir, sample_rate):
    files1 = get_audio_files_recursive(folder1)  # Using the recursive function for source 1
    files2 = get_audio_files(folder2)  # Using the original function for source 2

    # Shuffle the lists to ensure random selection
    random.shuffle(files1)
    random.shuffle(files2)

    os.makedirs(output_dir, exist_ok=True)

    for file2 in tqdm(files2, desc="Processing Files"):
        success = False
        while files1 and not success:
            file1 = files1.pop(random.randint(0, len(files1) - 1))
            success = process_audio(file1, file2, output_dir, sample_rate)


def main():
    parser = argparse.ArgumentParser(description="Mix audios from two folders.")

    parser.add_argument("--folder1", required=True, help="Path to the first folder containing audio files.")
    parser.add_argument("--folder2", required=True, help="Path to the second folder containing audio files.")
    parser.add_argument("--output_dir", required=True, help="Path to the directory where mixed audio files will be saved.")
    parser.add_argument("--sample_rate", type=int, required=False, default=16000, help="Sample rate to use for audio processing (default is 16000 Hz).")
    
    args = parser.parse_args()

    mix_audios(args.folder1, args.folder2, args.output_dir, args.sample_rate)
    print(f"Audio mixing complete. Mixed files saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
