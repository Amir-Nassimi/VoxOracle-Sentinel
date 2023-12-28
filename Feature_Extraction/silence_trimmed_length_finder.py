import os
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


def remove_silence(audio, silence_thresh, remove_edge_silence):
    if remove_edge_silence:
        non_silent_parts = detect_nonsilent(audio, silence_thresh=silence_thresh)
        if non_silent_parts:
            start, end = non_silent_parts[0][0], non_silent_parts[-1][1]
            return audio[start:end]
        else:
            return AudioSegment.silent(duration=0)
    else:
        segments = split_on_silence(audio, silence_thresh=silence_thresh)
        if segments:
            return segments[0]
        else:
            return AudioSegment.silent(duration=0)


def process_file(file_path, silence_thresh, remove_edge_silence):
    audio = AudioSegment.from_wav(file_path)
    audio_processed = remove_silence(audio, silence_thresh, remove_edge_silence)
    return audio_processed


def get_lengths(directory_path, silence_thresh, remove_edge_silence):
    lengths = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            audio_processed = process_file(file_path, silence_thresh, remove_edge_silence)
            lengths.append(len(audio_processed))

    return lengths


def calculate_max_and_mean(lengths):
    if lengths:
        max_length = max(lengths)
        mean_length = sum(lengths) / len(lengths)
        return max_length, mean_length
    else:
        return 0, 0


def main():
    parser = argparse.ArgumentParser(description="Process audio files and calculate maximum and mean length in milliseconds.")
    parser.add_argument("--slc_thr", required=False, type=float, default=-50, help="Silence threshold in dBFS")
    parser.add_argument("--directory_path", required=True, type=str, help="Path to the directory containing audio files")
    parser.add_argument("--remove_edge_silence", required=False, type=bool, default=True, help="Remove silence from the beginning and end of audio files before measuring length")
    args = parser.parse_args()
    
    lengths = get_lengths(args.directory_path, args.slc_thr, args.remove_edge_silence)
    max_length, mean_length = calculate_max_and_mean(lengths)

    print(f"The mean length of audio files in the directory is {mean_length} milliseconds.")
    print(f"The maximum length of audio files in the directory is {max_length} milliseconds.")


if __name__ == "__main__":
    main()
