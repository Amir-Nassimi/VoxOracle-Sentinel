import os
import random
import librosa
import argparse
from tqdm import tqdm
from glob import glob
import soundfile as sf
from singleton_decorator import singleton


@singleton
class AudioMixer:
    def __init__(self, folder1, folder2, output_dir, sample_rate):
        self.folder1 = folder1
        self.folder2 = folder2
        self.output_dir = output_dir
        self.sample_rate = sample_rate

    @staticmethod
    def get_audio_files(directory, recursive=False):
        if recursive:
            return glob(os.path.join(directory, '**', '*.wav'), recursive=True)
        else:
            return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]

    def process_audio(self, file1, file2):
        audio1, sr1 = librosa.load(file1, sr=self.sample_rate, mono=True)
        audio2, _ = librosa.load(file2, sr=self.sample_rate, mono=True)

        len1, len2 = len(audio1), len(audio2)
        if len2 <= len1:
            start = random.randint(0, len1 - len2)
            audio1 = audio1[start:start + len2]
        else:
            return False

        mixed_audio = audio1 + audio2
        relative_path = os.path.relpath(file2, self.folder2)
        output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))
        os.makedirs(output_subdir, exist_ok=True)

        output_filename = os.path.basename(file1).replace('.wav', '_mixed.wav')
        output_path = os.path.join(output_subdir, output_filename)
        sf.write(output_path, mixed_audio, sr1)
        return True

    def mix_audios(self):
        files1 = self.get_audio_files(self.folder1, recursive=True)
        files2 = self.get_audio_files(self.folder2)

        random.shuffle(files1)
        random.shuffle(files2)
        os.makedirs(self.output_dir, exist_ok=True)

        for file2 in tqdm(files2, desc="Processing Files"):
            success = False
            while files1 and not success:
                file1 = files1.pop(random.randint(0, len(files1) - 1))
                success = self.process_audio(file1, file2)


def main():
    parser = argparse.ArgumentParser(description="Mix audios from two folders.")
    parser.add_argument("--folder1", required=True, help="Path to the first folder containing audio files.")
    parser.add_argument("--folder2", required=True, help="Path to the second folder containing audio files.")
    parser.add_argument("--output_dir", required=True,
                        help="Path to the directory where mixed audio files will be saved.")
    parser.add_argument("--sample_rate", type=int, required=False, default=16000,
                        help="Sample rate to use for audio processing (default is 16000 Hz).")

    args = parser.parse_args()

    audio_mixer = AudioMixer(args.folder1, args.folder2, args.output_dir, args.sample_rate)
    audio_mixer.mix_audios()
    print(f"Audio mixing complete. Mixed files saved in '{args.output_dir}'.")


if __name__ == "__main__":
    main()
