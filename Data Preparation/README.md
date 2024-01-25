# VoxOracle Sentinel

## Dataset Preparation

Organize your dataset with labeled speech command samples. The dataset structure should include audio files and corresponding labels. the Dataset itself can be found in the main directory of this branch.

### Volume Based Audio Mover

It is designed to analyze and move audio files based on their average volume level. Here's a summary of its functionality:

- ***Volume Threshold Analysis***: The script evaluates audio files to determine if their average volume level (in decibels) is below a specified threshold. This threshold is provided by the user as a command-line argument (`--dB`).

- ***File Processing and Movement***: Audio files that have an average volume below the threshold are moved from a source directory to a destination directory. Both directories are specified by the user through command-line arguments (`--original_folder` and `--destination_folder`).

- ***Directory Handling***: The script creates the destination directory if it does not already exist. It then iterates through all files in the source directory, skipping any subdirectories it encounters.

- ***Audio File Handling***: For each audio file, the script uses the `pydub` library to calculate its average decibel level. If this level is below the provided threshold, the script moves the file to the destination directory.

- ***Command-Line Interface***: Users interact with the script through a command-line interface, providing the necessary arguments for decibel threshold, source directory, and destination directory.

- ***Outcome Reporting***: After processing, the script reports the total number of audio files that were moved to the destination directory.

This script is particularly useful for sorting or organizing audio files based on their loudness, such as in audio database management, where files may need to be categorized or filtered based on their volume levels.

```bash
python volume_based_audio_mover.py --original_folder [input data dir] --destination_folder [output data dir] --dB[input threshold]
```
---

### Short Audio Counter

To count the number of audio files shorter than or equal to a specified duration threshold, use the following command:
```bash
python short_audio_counter.py.py [/path/to/directory1] [/path/to/directory2] --threshold [input length]
```
---

### Dataset Splitter

It is designed for splitting an audio dataset into training, validation, and testing sets. It also generates corresponding CSV files for each set. Here's a summary of its functionality:

- ***split_dataset***: This function takes in paths to the dataset directory, training, validation, and testing directories, and paths for saving corresponding CSV files. It also accepts `valid_split` and `test_split` parameters that define the fraction of data to be used for validation and testing, respectively.

- ***Creating Directories***: The function creates directories for training, validation, and testing data if they don't already exist.

- ***Processing and Splitting Data***: It processes each label in the dataset directory and splits the files into training, validation, and testing sets based on the provided split ratios. The files are randomly shuffled to ensure randomness in the split.

- ***Copying Files and Generating CSVs***: The script copies files to their respective directories (training, validation, testing) and records their paths and labels in a list. These lists are then used to create CSV files that map each file to its label, aiding in the dataset's management.

In summary, this script is a useful tool for preprocessing audio datasets, particularly for machine learning tasks where organized and labeled data is crucial. It automates the process of splitting the dataset and creating useful metadata in the form of CSV files.

```bash
python dataset_split.py --train_dir [path_to_training_dir] --valid_dir [path_to_validation_dir] --test_dir [path_to_testing_dir] --train_csv [path_to_training_csv] --valid_csv [path_to_validation_csv] --test_csv [path_to_testing_csv] --dataset_dir [path_to_dataset] --valid_split [default value: 0.1] --test_split [default value: 0.1]
```
---

### Random Audio Mixer

It is designed to mix audio files from two different folders, creating a new set of audio files that combine elements from both sources.

- ***File Retrieval***: It includes a static method `get_audio_files` to retrieve `.wav` files from the specified directories. This method can operate either recursively or non-recursively.

- ***Audio Processing***: The `process_audio` method handles the combination of audio files from the two sources, adjusting their lengths to match and then mixing them.

- ***Mixing Workflow***: The `mix_audios` method orchestrates the entire process, fetching files from both sources, shuffling them for randomness, and then processing each pair of files.

This script is particularly useful for applications in audio data augmentation, machine learning model training, or any scenario where blending different audio sources is beneficial. The OOP design makes the script modular and easily extendable for future enhancements or adaptations.

```bash
python audio_mixer.py --folder1 [/path/to/first/folder] --folder2 [/path/to/second/folder] --output_dir [/path/to/output/folder] --sample_rate [default value: 16000]
```