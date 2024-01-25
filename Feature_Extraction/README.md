# VoxOracle Sentinel

## Feature Extraction

### Silence Trimmed Length Finder

It is designed to process audio files and determine the maximum length of these files after optional silence removal. Here's a summary of its functionality:

- ***Silence Removal Options***: The script can either remove silence from just the beginning of each audio file or from both the beginning and end. This is controlled by the `remove_edge_silence` parameter, which, when set to `True`, activates the removal of silence from both edges of the audio. The level of silence to be removed is defined by a 'silence threshold' (`silence_thresh`), adjustable by the user.

- ***Audio Processing***: Each `.wav` audio file in the specified directory is processed to remove silence based on the user's choice. The script uses the `pydub` library's `split_on_silence` function to handle silence removal from the beginning, and `detect_nonsilent` for removing silence from both edges.

- ***Maximum and Mean Length Calculation***: After processing each file, the script measures its length in milliseconds. It then compares this length to the lengths of previously processed files, keeping track of the longest and the mean of all audio files encountered.

- ***Command-Line Interface***: Users interact with the script through a command-line interface. The script uses `argparse` for handling command-line arguments, allowing users to specify the directory containing the audio files (`--directory_path`), the silence threshold (`--slc_thr`), and whether to remove silence from just the beginning or both edges of the audio files (`--remove_edge_silence`).

- ***Output***: After processing all files in the specified directory, the script outputs the length of the longest audio file (post-processing) in milliseconds.

This script is particularly useful for audio analysis tasks where understanding the length of audio files post-processing is important, such as in preparing datasets for machine learning or in audio editing projects where uniform audio length is desired.

```bash
python silence_trimmed_length_finder.py --directory_path --slc_thr [-50] --remove_edge_silence [True]
```
---

### Feature Extractor (MelSpectogram)

The provided Python script is an audio feature extraction tool, primarily designed to process audio files and extract their Mel Spectrogram features. It uses libraries like `os`, `librosa`, `pathlib`, `argparse`, `numpy`, `pandas`, and `tqdm` to handle audio files and perform data processing. Here's a summary of its key features and functionalities:

1. **AudioFeatureExtractor Class**:
   - **Initialization (__init__)**: Sets up the class with parameters:
     - `audio_dir`: Directory containing audio files.
     - `save_numpy_dir`: Directory to save extracted features in numpy format.
     - `save_pickle_path`: Directory to save a DataFrame of the features in pickle format.
     - `target_len`: Target length for audio files.
     - `sample_rate`: Sample rate for audio processing, default set to 16000.
     - `n_mels`: Number of Mel bands to use, default set to 128.
   - **standardize_audio_length**: Static method to adjust the length of the audio to the target length.
   - **extract_mel_spectrogram**: Extracts Mel Spectrogram from an audio file or numpy array.
     - Converts audio to a Mel Spectrogram.
     - Squares the amplitude-to-decibel converted spectrogram.
     - Replicates it across three channels to simulate an RGB image format.

2. **Extract Features**:
   - Iterates over all `.wav` files in the given `audio_dir`.
   - Extracts Mel Spectrogram for each audio file.
   - Saves the spectrogram as a numpy array in `save_numpy_dir`.
   - Compiles a list of file paths and corresponding labels.
   - Saves this data as a pandas DataFrame in a pickle file.

3. **Command-Line Interface**:
   - Uses `argparse` to define and parse command-line arguments.
   - Users can specify the audio directory, numpy files directory, pickle file path, and target audio length.

4. **Batch Processing with Progress Visualization**:
   - Uses `tqdm` to display a progress bar while processing files.
   - Processes each `.wav` file in the specified directory, handling them in a batch manner.

5. **DataFrame and Data Persistence**:
   - Compiles the paths to the numpy files and their labels into a pandas DataFrame.
   - Saves this DataFrame in a pickle file for easy retrieval and use in other applications, like machine learning.

In summary, this script is a specialized tool for extracting Mel Spectrogram features from audio files, useful for tasks like audio analysis, machine learning, and signal processing where such features are relevant. The script facilitates batch processing of audio files and organizes the extracted features in a structured and easily accessible format.

```bash
python your_script_name.py --audio_dir /path/to/audio_files --numpy_dir /path/to/numpy_files --pickle_path /path/to/save/pickle --target_len [2.0]
```