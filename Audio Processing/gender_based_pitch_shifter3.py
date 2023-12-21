import os
import librosa
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf
from inaSpeechSegmenter import Segmenter
from singleton_decorator import singleton
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


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
        self.audio_files, self.male_sffs, self.female_sffs = [], [], []

    def analyze_directory(self):
        for filename in tqdm(os.listdir(self.input_dir), desc="Analyzing Audio Files"):
            if filename.endswith(".wav"):
                self.analyze_file(filename)
        
        male_distribution = np.array(self.male_sffs).reshape(-1, 1)
        female_distribution = np.array(self.female_sffs).reshape(-1, 1)

        male_stats = self._fit_gaussian_mixture(male_distribution)
        female_stats = self._fit_gaussian_mixture(female_distribution)

        return male_stats, female_stats

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

                if label == 'male':
                    self.male_sffs.append(sff)
                elif label == 'female':
                    self.female_sffs.append(sff)

                audio_data.sff_data.append((label, sff, start, end))

        self.audio_files.append(audio_data)

    def calculate_sff(self, audio):
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch = np.max(pitches)
        return pitch

    @staticmethod
    def _fit_gaussian_mixture(distribution, max_components=10):
        # Standardize the distribution
        scaler = StandardScaler()
        distribution_scaled = scaler.fit_transform(distribution)

        no = 0
        best_gmm = None
        lowest_bic = np.infty
        lowest_aic = np.infty

        # Test different numbers of components
        for n_components in tqdm(range(1, max_components + 1), desc="Fitting GMMs"):
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(distribution_scaled)
            bic = gmm.bic(distribution_scaled)
            aic = gmm.aic(distribution_scaled)

            if bic < lowest_bic:
                no = n_components
                lowest_bic = bic
                best_gmm = gmm
            if aic < lowest_aic:
                no = n_components
                lowest_aic = aic
                best_gmm = gmm

        print(f"best gmm has been found with {no} components")
        # Extract statistics from the best GMM model
        stats = {
            'mean': scaler.inverse_transform([best_gmm.means_[0]])[0][0],
            'std': scaler.scale_[0]
        }

        return stats

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
        self.male_mean = male_stats['mean']
        self.male_std = male_stats['std']
        self.female_mean = female_stats['mean']
        self.female_std = female_stats['std']

    def calculate_shift_amount(self, gender, sff, aug_index):
        # Base shift calculation remains the same
        target_mean = self.female_mean if gender == 'male' else self.male_mean
        target_std = self.female_std if gender == 'male' else self.male_std
        basic_shift = target_mean - sff
        shift_amount = basic_shift * (target_std / self.male_std if gender == 'male' else target_std / self.female_std)

        # Introduce variability based on augmentation index
        multiplier = 1 + (aug_index * 0.2)  # 20% increase per augmentation
        varied_shift = shift_amount * multiplier

        print(varied_shift)

        return 5

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

        for i in tqdm(range(1, self.num_augmentations+1, 1), desc="Process Files"):
            processed_audio = []
            for label, sff, start, end in audio_data.sff_data:
                if label in ['male', 'female']:
                    shift_amount = self.shifter.calculate_shift_amount(label, sff, i)
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
    parser.add_argument("--num_augmentations", required=False, type=int, default=1, help="Number of augmentations per audio file.")
    parser.add_argument("--thrsh", required=False, action='store_true', help="To completely augment (value:True) or only change the part (value:False) in which the speaker speaks.")
    
    args = parser.parse_args()

    analyzer = SFFAnalyzer(args.input_dir, args.sample_rate, args.thrsh)  # Assuming a fixed sample rate for simplicity
    male_stats, female_stats = analyzer.analyze_directory()

    executer = Executer(male_stats, female_stats, **vars(args))
    executer.process_directory(analyzer)

if __name__ == "__main__":
    main()