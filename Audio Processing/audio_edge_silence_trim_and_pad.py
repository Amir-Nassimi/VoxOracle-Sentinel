import os
import argparse
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def remove_silence_from_edges(audio, silence_thrsh):
    # Detect non-silent parts
    nonsilent_parts = detect_nonsilent(audio, silence_thresh=silence_thrsh)
    if nonsilent_parts:
        start, end = nonsilent_parts[0][0], nonsilent_parts[-1][1]
        return audio[start:end]
    else:
        return AudioSegment.silent(duration=0)

def add_padding(audio, target_length):
    # Calculate total padding needed
    total_padding = max(0, target_length - len(audio))
    # Divide padding to add it equally at the beginning and end
    padding_each_side = total_padding // 2
    # Create padding audio
    padding = AudioSegment.silent(duration=padding_each_side)
    # Add padding to both sides
    return padding + audio + padding

def process_file(file_path, target_length, silence_thrsh):
    audio = AudioSegment.from_wav(file_path)
    
    # Process only if audio length is less than or equal to target_length
    if len(audio) <= target_length:
        audio_without_edges = remove_silence_from_edges(audio, silence_thrsh)
        padded_audio = add_padding(audio_without_edges, target_length)
        return padded_audio
    else:
        return None

def process_directory(directory_path, target_length, silence_thrsh, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory_path, filename)
            padded_audio = process_file(file_path, target_length, silence_thrsh)

            if padded_audio is not None:
                # Save the padded audio in the specified output directory
                padded_file_path = os.path.join(output_dir, f"padded_{filename}")
                padded_audio.export(padded_file_path, format="wav")

def main():
    parser = argparse.ArgumentParser(description="Process audio files: remove edge silence, pad to target length.")
    parser.add_argument("--slc_thr", required=False, type=float, default=-50, help="Silence threshold in dBFS for edge silence removal")
    parser.add_argument("--target_length", required=True, type=int, help="Desired final length of each audio file in milliseconds")
    parser.add_argument("--directory_path", required=True, type=str, help="Path to the directory containing audio files")
    parser.add_argument("--output_dir", required=True, type=str, help="Path to the directory where processed audio files will be saved")
    args = parser.parse_args()

    process_directory(args.directory_path, args.target_length, args.slc_thr, args.output_dir)
    print(f"Processing complete. Processed audio files saved in '{args.output_dir}'.")

if __name__ == "__main__":
    main()