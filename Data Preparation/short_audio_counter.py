import os
import argparse
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def remove_silence_from_edges(audio_path, silence_thrsh):
    audio = AudioSegment.from_file(audio_path, format="wav")
    nonsilent_parts = detect_nonsilent(audio, silence_thresh=silence_thrsh)
    if nonsilent_parts:
        start, end = nonsilent_parts[0][0], nonsilent_parts[-1][1]
        return audio[start:end]
    else:
        return AudioSegment.silent(duration=0)


def count_short_audios(input_dirs, threshold_length, silence_thrsh):
    short_audios = 0

    for input_dir in tqdm(input_dirs, desc='Analyzing directories'):
        for root, _, files in os.walk(input_dir):
            for file in files:
                audio_path = os.path.join(root, file)
                try:
                    # Remove silence from edges
                    trimmed_audio = remove_silence_from_edges(audio_path, silence_thrsh)
                    duration_in_milliseconds = len(trimmed_audio)

                    if duration_in_milliseconds <= threshold_length:
                        short_audios += 1
                except Exception as e:
                    print(f"Error processing {audio_path}: {str(e)}")

    return short_audios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count short audio files.")

    parser.add_argument("input_dirs", nargs="+", help="Input directories containing audio files.")
    parser.add_argument("--threshold", type=float, required=True, help="Threshold length in milliseconds.")
    parser.add_argument("--silence_thrsh", type=float, required=False, default=-50.0, help="Silence threshold in dBFS for edge silence removal.")
    args = parser.parse_args()

    input_dirs = args.input_dirs
    threshold_length = args.threshold
    silence_thrsh = args.silence_thrsh

    short_audios = count_short_audios(input_dirs, threshold_length, silence_thrsh)
    print(f"Number of audio files with duration less than {threshold_length} milliseconds after silence removal: {short_audios}")