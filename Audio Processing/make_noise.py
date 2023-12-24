import os
import glob
import librosa
import argparse
import numpy as np
import soundfile as sf
from multiprocessing import Pool

class AudioProcessor:
    def __init__(self, input_dirs, output_base_dir, input_length, sample_rate, noise_factor):
        self.input_dirs = input_dirs
        self.output_base_dir = output_base_dir
        self.input_length = input_length
        self.sample_rate = sample_rate
        self.noise_factor = noise_factor

    def get_output_path(self, input_path):
        # Find the base input directory that is a parent of the input path
        for input_dir in self.input_dirs:
            if input_path.startswith(input_dir):
                # Replace the base input directory with the base output directory
                relative_path = os.path.relpath(input_path, input_dir)
                return os.path.join(self.output_base_dir, relative_path)
        # Default case if no matching input directory found
        return os.path.join(self.output_base_dir, os.path.basename(input_path))

    def process_single_audio(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # Convert input_length from milliseconds to seconds, then to samples
            desired_length_in_samples = int((self.input_length / 1000.0) * self.sample_rate)

            if len(audio) < desired_length_in_samples:
                total_padding = desired_length_in_samples - len(audio)
                padding_start = total_padding // 2
                padding_end = total_padding - padding_start
                audio = np.pad(audio, (padding_start, padding_end), 'constant')
            else:
                # Crop the audio to the desired length
                audio = audio[:desired_length_in_samples]

            noisy_audio = audio * self.noise_factor
            output_path = self.get_output_path(audio_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, noisy_audio, sr)
            print(f"Processed: {audio_path}")

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")

    def process_audio(self):
        pool = Pool()
        audio_paths = []
        for input_dir in self.input_dirs:
            audio_paths.extend(glob.glob(os.path.join(input_dir, '**', '*.wav'), recursive=True))

        pool.map(self.process_single_audio, audio_paths)
        pool.close()
        pool.join()

def main():
    parser = argparse.ArgumentParser(description="Audio processing script")
    parser.add_argument("--input_dirs", nargs='+', required=True, help="Input directories containing audio files.")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Base output directory for processed audio.")
    parser.add_argument("--input_length", type=float, required=False, help="Desired input length in seconds.")
    parser.add_argument("--sample_rate", type=int, required=False, default=16000, help="Sample rate for audio.")
    parser.add_argument("--noise_factor", type=float, default=0.1, help="Multiplication factor for noise.")
    args = parser.parse_args()

    audio_processor = AudioProcessor(args.input_dirs, args.output_base_dir, args.input_length, args.sample_rate, args.noise_factor)
    audio_processor.process_audio()

if __name__ == "__main__":
    main()