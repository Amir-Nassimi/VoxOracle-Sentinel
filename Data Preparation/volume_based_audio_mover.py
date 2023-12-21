import os
import shutil
import argparse
from pydub import AudioSegment

def move_files_to_directory(src_directory, dest_directory, threshold_dB):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    # Counter for the total number of songs moved
    total_songs_moved = 0

    # Iterate through all files in the source directory
    for filename in os.listdir(src_directory):
        filepath = os.path.join(src_directory, filename)

        # Check if the file is a directory (skip directories)
        if os.path.isdir(filepath):
            continue

        # Check if the voice is quieter than the threshold
        if is_quiet(filepath, threshold_dB):
            dest_filepath = os.path.join(dest_directory, filename)
            shutil.move(filepath, dest_filepath)
            total_songs_moved += 1
            #print(f"Moved {filename} to {dest_directory}")

    print(f"\n{src_directory} - Total sounds moved: {total_songs_moved}")
    

def is_quiet(filepath, threshold_dB):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(filepath)

    # Calculate the average volume level in decibels
    average_dB = audio.dBFS

    # Check if the average volume is below the threshold
    return average_dB < threshold_dB


if __name__ == "__main__":
    # Set the source and destination directories
    parser = argparse.ArgumentParser(description="Check Duplicat Data from same person and Move them")

    parser.add_argument("--dB", type=float, required=True,default=-47, help="number of decibel")
    parser.add_argument("--original_folder", type=str, required=True, help="Path to original folder")
    parser.add_argument("--destination_folder", type=str, required=False, help="Path to destination folder")

    arg = parser.parse_args()

    threshold = arg.dB
    source_directory = arg.original_folder
    destination_directory = arg.destination_folder

    move_files_to_directory(source_directory, destination_directory, threshold)