import os
import librosa
import argparse
import numpy as np
import soundfile as sf
from inaSpeechSegmenter import Segmenter
from singleton_decorator import singleton


@singleton
class GenderDetector:
    def __init__(self):
        self.segmenter = Segmenter()

    def detect_genders(self, audio_path):
        return self.segmenter(audio_path)

class AudioFileProcessor:
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.audio, self.sample_rate = librosa.load(file_path, sr=sample_rate)

    def apply_pitch_shift(self, start, end, shift, thrsh):
        print(f"Applying pitch shift: Start: {start}, End: {end}, Shift: {shift}, Threshold: {thrsh}")
        if thrsh: segment = self.audio[int(self.sample_rate * start):int(self.sample_rate * end)]
        else: segment = self.audio

        shifted_segment = librosa.effects.pitch_shift(y=segment, sr=self.sample_rate, n_steps=shift)
        return shifted_segment

@singleton
class PitchShifter:
    def __init__(self, min_shift, max_shift, male_sff, female_sff):
        self.min_shift = min_shift
        self.max_shift = max_shift

        self.male_sff = male_sff
        self.female_sff = female_sff

    def calculate_sff(self, audio, sr):
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch = np.max(pitches)
        return pitch

    def get_unique_shift_amount_sff(self, gender, sff, aug_index, num_augmentations):
        print(f"Calculating shift amount: Gender: {gender}, SFF: {sff}, Index: {aug_index}, Num Augmentations: {num_augmentations}")
        if gender == 'male' and sff >= self.male_sff:
            shift_range = np.linspace(self.min_shift, self.max_shift, num_augmentations)
        elif gender == 'female' and sff < self.female_sff:
            shift_range = np.linspace(-self.max_shift, -self.min_shift, num_augmentations)
        else:
            return 0  # No shift for androgynous or already within target range

        return shift_range[aug_index]
    
    def get_unique_shift_amount(self, gender, sff, aug_index, num_augmentations):
        print(f"Calculating shift amount: Gender: {gender}, SFF: {sff}, Index: {aug_index}, Num Augmentations: {num_augmentations}")
        if gender == 'male':
            shift_range = np.linspace(self.min_shift, self.max_shift, num_augmentations)
        elif gender == 'female':
            shift_range = np.linspace(-self.max_shift, -self.min_shift, num_augmentations)
        else:
            return 0  # No shift for androgynous or already within target range

        return shift_range[aug_index]


class Executer:
    def __init__(self, **kwargs):
        self.thrsh = kwargs.get("thrsh")
        self.input_dir = kwargs.get("input_dir")
        self.output_dir = kwargs.get("output_dir")
        self.sample_rate = kwargs.get("sample_rate")
        self.num_augmentations = kwargs.get("num_augmentations")

        self.use_sff = kwargs.get("use_sff")

        self.shifter = PitchShifter(kwargs.get("min_shift"), kwargs.get("max_shift"), kwargs.get("male_sff"), kwargs.get("female_sff"))
        self.detector = GenderDetector()

    def process_directory(self):
        for filename in os.listdir(self.input_dir):
            if filename.endswith(".wav"):
                self.process_file(filename)

    def process_file(self, filename):
        input_audio = os.path.join(self.input_dir, filename)
        processor = AudioFileProcessor(input_audio, self.sample_rate)

        for i in range(self.num_augmentations):
            processed_audio = []
            for segment in self.detector.detect_genders(input_audio):
                label, start, end = segment
                if label in ['male', 'female']:
                    if self.thrsh: 
                        segment_audio = processor.audio[int(processor.sample_rate * start):int(processor.sample_rate * end)]
                    else:
                        segment_audio = processor.audio
                    sff = self.shifter.calculate_sff(segment_audio, processor.sample_rate)

                    if self.use_sff: shift_amount = self.shifter.get_unique_shift_amount_sff(label, sff, i, self.num_augmentations)
                    else: shift_amount = self.shifter.get_unique_shift_amount(label, sff, i, self.num_augmentations)

                    shifted_segment = processor.apply_pitch_shift(start, end, shift_amount, self.thrsh)
                    processed_audio.append(shifted_segment)

            augmented_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.wav"
            output_file_path = os.path.join(self.output_dir, augmented_filename)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            try:
                sf.write(output_file_path, np.concatenate(processed_audio), processor.sample_rate)
                print(f"Augmented audio saved to {output_file_path}")
            except Exception as e:
                print(f"Failed to save {output_file_path}: {e}")
                

def main():
    parser = argparse.ArgumentParser(description="Audio processing with gender detection and pitch shifting.")
    
    parser.add_argument("--male_sff", required=False, type=float, default=130, help="Male SFF threshold.")
    parser.add_argument("--female_sff", required=False, type=float, default=180, help="Female SFF threshold.")
    parser.add_argument("--use_sff", required=False, action='store_false', help="Do not use SFF values for pitch shifting.")

    parser.add_argument("--output_dir", required=True, help="Output directory for processed audio.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing audio files.")
    parser.add_argument("--min_shift", required=False, type=float, default=-3, help="Minimum pitch shift factor.")
    parser.add_argument("--max_shift", required=False, type=float, default=3.5, help="Maximum pitch shift factor.")
    parser.add_argument("--sample_rate", required=False, type=int, default=16000, help="Sample rate for audio processing.")
    parser.add_argument("--num_augmentations", required=False, type=int, default=1, help="Number of augmentations per audio file.")
    parser.add_argument("--thrsh", required=False, action='store_false', help="To Completely Augment (value:False) or Only change the part (value:True) in which the speaker speaks.")

    args = parser.parse_args()

    executer = Executer(**vars(args))
    executer.process_directory()

if __name__ == "__main__":
    main()