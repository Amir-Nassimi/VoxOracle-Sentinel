import os
import tqdm
import argparse
import librosa
import numpy as np
import soundfile as sf
from inaSpeechSegmenter import Segmenter
from singleton_decorator import singleton

class AudioData:
    def __init__(self, filename, audio, sample_rate):
        self.filename = filename
        self.audio = audio
        self.sample_rate = sample_rate
        self.sff_data = []

@singleton
class GenderDetector:
    def __init__(self):
        self.segmenter = Segmenter()

    def detect_genders(self, audio_path):
        return self.segmenter(audio_path)

class SFFAnalyzer:
    def __init__(self, input_dir, sample_rate, thrsh):
        self.thrsh = thrsh
        self.input_dir = input_dir
        self.sample_rate = sample_rate
        self.audio_files = []

    def analyze_directory(self):
        for filename in tqdm(os.listdir(self.input_dir), desc="Analyzing Audio Files"):
            if filename.endswith(".wav"):
                self.analyze_file(filename)

        return self.calculate_stats()

    def analyze_file(self, filename):
        file_path = os.path.join(self.input_dir, filename)
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        audio_data = AudioData(filename, audio, self.sample_rate)

        for segment in GenderDetector().detect_genders(file_path):
            label, start, end = segment
            if label in ['male', 'female']:
                
                if not self.thrsh: segment_audio = audio[int(self.sample_rate * start):int(self.sample_rate * end)]
                else: segment_audio = audio

                sff = self.calculate_sff(segment_audio)
                audio_data.sff_data.append((label, sff, start, end))

        self.audio_files.append(audio_data)

    def calculate_sff(self, audio):
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch = np.max(pitches)
        return pitch

    def calculate_stats(self):
        male_sffs = []
        female_sffs = []
        for audio_data in self.audio_files:
            for gender, sff, _, _ in audio_data.sff_data:
                if gender == 'male':
                    male_sffs.append(sff)
                elif gender == 'female':
                    female_sffs.append(sff)

        male_stats = self._calculate_stats(male_sffs)
        female_stats = self._calculate_stats(female_sffs)

        return male_stats, female_stats

    @staticmethod
    def _calculate_stats(sff_values):
        if not sff_values:  # Handle empty lists
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        return {
            'min': min(sff_values),
            'max': max(sff_values),
            'mean': np.mean(sff_values),
            'median': np.median(sff_values)
        }

class AudioFileProcessor:
    def __init__(self, audio_data):
        self.audio_data = audio_data

    def apply_pitch_shift(self, start, end, shift, thrsh):
        print(f"Applying pitch shift: Start: {start}, End: {end}, Shift: {shift}, Threshold: {thrsh}")
        if not thrsh:
            segment = self.audio_data.audio[int(self.audio_data.sample_rate * start):int(self.audio_data.sample_rate * end)]
        else:
            segment = self.audio_data.audio

        shifted_segment = librosa.effects.pitch_shift(segment, sr=self.audio_data.sample_rate, n_steps=shift)
        return shifted_segment

@singleton
class PitchShifter:
    def __init__(self, male_stats, female_stats):
        self.male_stats = male_stats
        self.female_stats = female_stats

    def calculate_shift_amount(self, gender, sff):
        SCALE_FACTOR = 0.1  # Adjust based on testing

        # Select appropriate stats for current and target genders
        current_stats = self.male_stats if gender == 'male' else self.female_stats
        target_stats = self.female_stats if gender == 'male' else self.male_stats

        # Normalize current SFF
        normalized_sff = (sff - current_stats['min']) / (current_stats['max'] - current_stats['min'])

        # Normalize target SFF (mean of opposite gender)
        normalized_target = (target_stats['median'] - target_stats['min']) / (target_stats['max'] - target_stats['min'])

        # Calculate shift amount and apply scaling factor
        shift_amount = (normalized_target - normalized_sff) * SCALE_FACTOR

        return shift_amount

class Executer:
    def __init__(self, male_stats, female_stats, **kwargs):
        self.thrsh = kwargs.get("thrsh")
        self.output_dir = kwargs.get("output_dir")
        self.num_augmentations = kwargs.get("num_augmentations")
        self.shifter = PitchShifter(male_stats, female_stats)

    def process_directory(self, analyzer):
        for audio_data in tqdm(analyzer.audio_files, desc="Processing Audio Files"):
            self.process_file(audio_data)

    def process_file(self, audio_data):
        processor = AudioFileProcessor(audio_data)

        for i in range(self.num_augmentations):
            processed_audio = []
            for label, sff, start, end in audio_data.sff_data:
                if label in ['male', 'female']:
                    shift_amount = self.shifter.calculate_shift_amount(label, sff)
                    shifted_segment = processor.apply_pitch_shift(start, end, shift_amount, self.thrsh)
                    processed_audio.append(shifted_segment)

            augmented_filename = f"{os.path.splitext(audio_data.filename)[0]}_aug_{i}.wav"
            output_file_path = os.path.join(self.output_dir, augmented_filename)

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            try:
                sf.write(output_file_path, np.concatenate(processed_audio), audio_data.sample_rate)
                print(f"Augmented audio saved to {output_file_path}")
            except Exception as e:
                print(f"Failed to save {output_file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Audio processing with gender detection and pitch shifting.")
    
    parser.add_argument("--output_dir", required=True, help="Output directory for processed audio.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing audio files.")
    parser.add_argument("--sample_rate", required=False, type=int, default=16000, help="Sample rate for audio processing.")
    parser.add_argument("--thrsh", required=False, action='store_true', help="To completely augment (value:True) or only change the part (value:False) in which the speaker speaks.")
    parser.add_argument("--num_augmentations", required=False, type=int, default=1, help="Number of augmentations per audio file.")

    args = parser.parse_args()

    analyzer = SFFAnalyzer(args.input_dir, args.sample_rate, args.thrsh)  # Assuming a fixed sample rate for simplicity
    male_stats, female_stats = analyzer.analyze_directory()

    executer = Executer(male_stats, female_stats, **vars(args))
    executer.process_directory(analyzer)

if __name__ == "__main__":
    main()