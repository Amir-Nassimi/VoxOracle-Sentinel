1. **Dataset Preparation:**
   Organize your dataset with labeled speech command samples. The dataset structure should include audio files and corresponding labels. the Dataset itself can be found in the main directory of this branch.

    1.1. ***Volume Based Audio Mover***

      It is designed to analyze and move audio files based on their average volume level. Here's a summary of its functionality:

      1. **Volume Threshold Analysis**: The script evaluates audio files to determine if their average volume level (in decibels) is below a specified threshold. This threshold is provided by the user as a command-line argument (`--dB`).

      2. **File Processing and Movement**: Audio files that have an average volume below the threshold are moved from a source directory to a destination directory. Both directories are specified by the user through command-line arguments (`--original_folder` and `--destination_folder`).

      3. **Directory Handling**: The script creates the destination directory if it does not already exist. It then iterates through all files in the source directory, skipping any subdirectories it encounters.

      4. **Audio File Handling**: For each audio file, the script uses the `pydub` library to calculate its average decibel level. If this level is below the provided threshold, the script moves the file to the destination directory.

      5. **Command-Line Interface**: Users interact with the script through a command-line interface, providing the necessary arguments for decibel threshold, source directory, and destination directory.

      6. **Outcome Reporting**: After processing, the script reports the total number of audio files that were moved to the destination directory.

      This script is particularly useful for sorting or organizing audio files based on their loudness, such as in audio database management, where files may need to be categorized or filtered based on their volume levels.

      ```bash
      cd ./Data Preparation
      python3 volume_based_audio_mover.py --original_folder [input data dir] --destination_folder [output data dir] --dB[input threshold]
      ```
      ---

    1.2. ***Silence Trimmed Length Finder***
      It is designed to process audio files and determine the maximum length of these files after optional silence removal. Here's a summary of its functionality:

      1. **Silence Removal Options**: The script can either remove silence from just the beginning of each audio file or from both the beginning and end. This is controlled by the `remove_edge_silence` parameter, which, when set to `True`, activates the removal of silence from both edges of the audio. The level of silence to be removed is defined by a 'silence threshold' (`silence_thresh`), adjustable by the user.

      2. **Audio Processing**: Each `.wav` audio file in the specified directory is processed to remove silence based on the user's choice. The script uses the `pydub` library's `split_on_silence` function to handle silence removal from the beginning, and `detect_nonsilent` for removing silence from both edges.

      3. **Maximum and Mean Length Calculation**: After processing each file, the script measures its length in milliseconds. It then compares this length to the lengths of previously processed files, keeping track of the longest and the mean of all audio files encountered.

      4. **Command-Line Interface**: Users interact with the script through a command-line interface. The script uses `argparse` for handling command-line arguments, allowing users to specify the directory containing the audio files (`--directory_path`), the silence threshold (`--slc_thr`), and whether to remove silence from just the beginning or both edges of the audio files (`--remove_edge_silence`).

      5. **Output**: After processing all files in the specified directory, the script outputs the length of the longest audio file (post-processing) in milliseconds.

      This script is particularly useful for audio analysis tasks where understanding the length of audio files post-processing is important, such as in preparing datasets for machine learning or in audio editing projects where uniform audio length is desired.

      ```bash
      cd ./Feature Extraction
      python silence_trimmed_length_finder.py --directory_path --slc_thr [-50] --remove_edge_silence [True]
      ```
      ---

    1.3. ***Audio Edge Silence Trim And Pad***

      It is designed to process audio files in a specified directory by performing the following actions:

      1. **Remove Silence from Edges**: For each `.wav` audio file in the provided input directory, the script first removes silence from both the beginning and the end. It does this using the `detect_nonsilent` function from the `pydub` library, which identifies nonsilent segments of the audio based on a given silence threshold (in dBFS). The silence threshold is adjustable through the `--slc_thr` command-line argument.

      2. **Measure and Pad Audio**: After removing the edge silence, the script measures the length of the resulting audio. If this length is less than the desired `target_length` (specified in milliseconds via the `--target_length` argument), it adds zero-padding equally to both the beginning and end of the audio. This padding ensures that the final length of each processed audio file matches the specified `target_length`.

      3. **Save Processed Files**: The processed (silence-trimmed and padded) audio files are then saved in a specified output directory. This output directory is set using the `--output_dir` command-line argument. The script ensures that this directory exists (creating it if necessary) and saves each processed file there with a `padded_` prefix added to the original filename.

      4. **Command-Line Interface**: The script uses the `argparse` library to provide a command-line interface. This allows users to specify the input directory (`--directory_path`), the target length for processed files (`--target_length`), the silence threshold (`--slc_thr`), and the output directory (`--output_dir`).

      5. **Error Handling**: The script includes basic error handling to deal with potential issues in audio file processing. It outputs informative messages if any problems occur during the processing of individual files.

      Overall, this script is a useful tool for standardizing the length of a collection of audio files, particularly useful in scenarios where consistent timing is crucial, such as in machine learning datasets, audio editing, or batch processing for multimedia projects.

      ```bash
      cd ./Audio Processing
      python audio_edge_silence_trim_and_pad.py --target_length [4020] --directory_path  --output_dir
      ```
      ---

    1.4. ***Short Audio Counter***

      To count the number of audio files shorter than or equal to a specified duration threshold, use the following command:
      ```bash
      cd ./Data Preparation
      python short_audio_counter.py.py [/path/to/directory1] [/path/to/directory2] --threshold [input length]
      ```
      ---

2. **Data Augmentation**:
    In this script, aligning with the research on speaking fundamental frequency (SFF) and gender perception:

      1. **Pitch Shift Range**: The script allows for dynamic range settings through command-line arguments (`--min_shift` and `--max_shift`). This flexibility aids in aligning the pitch shifts within natural-sounding boundaries (e.g., above 180 Hz for a feminine pitch and below 130 Hz for a masculine pitch).

      2. **Subtlety in Modification**: The script's approach of applying pitch shifts within a reasonable range supports subtle and natural-sounding voice modifications, avoiding drastic alterations that might sound unnatural.

      3. **Gender-Specific Adjustment**: The `PitchShifter` class uses the SFF to determine the direction of the shift (higher for feminine, lower for masculine), ensuring the modifications are in line with general gender perceptions of voice pitch.

      4. **Customization**: The script allows for augmentation of either the whole audio or specific segments where the speaker speaks (controlled by `--thrsh`), providing customization based on client-specific needs and preferences.

      5. **Multiple Augmentations**: The ability to create multiple augmentations per file, each with a unique pitch shift, offers a range of variations, which can be particularly useful in applications like voice training or research.


      ```bash
      cd ./Data Preparation
      python script.py --input_dir  --output_dir  --min_shift [0.2] --max_shift [1.2] --num_augmentations [3] --thrsh [False]
      ```
      