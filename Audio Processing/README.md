# VoxOracle Sentinel

## Audio Processing

### Audio Edge Silence Trim And Pad

It is designed to process audio files in a specified directory by performing the following actions:

- ***Remove Silence from Edges***: For each `.wav` audio file in the provided input directory, the script first removes silence from both the beginning and the end. It does this using the `detect_nonsilent` function from the `pydub` library, which identifies nonsilent segments of the audio based on a given silence threshold (in dBFS). The silence threshold is adjustable through the `--slc_thr` command-line argument.

- ***Measure and Pad Audio***: After removing the edge silence, the script measures the length of the resulting audio. If this length is less than the desired `target_length` (specified in milliseconds via the `--target_length` argument), it adds zero-padding equally to both the beginning and end of the audio. This padding ensures that the final length of each processed audio file matches the specified `target_length`.

- ***Save Processed Files***: The processed (silence-trimmed and padded) audio files are then saved in a specified output directory. This output directory is set using the `--output_dir` command-line argument. The script ensures that this directory exists (creating it if necessary) and saves each processed file there with a `padded_` prefix added to the original filename.

- ***Command-Line Interface***: The script uses the `argparse` library to provide a command-line interface. This allows users to specify the input directory (`--directory_path`), the target length for processed files (`--target_length`), the silence threshold (`--slc_thr`), and the output directory (`--output_dir`).

- ***Error Handling***: The script includes basic error handling to deal with potential issues in audio file processing. It outputs informative messages if any problems occur during the processing of individual files.

Overall, this script is a useful tool for standardizing the length of a collection of audio files, particularly useful in scenarios where consistent timing is crucial, such as in machine learning datasets, audio editing, or batch processing for multimedia projects.

```bash
python audio_edge_silence_trim_and_pad.py --target_length [4020] --directory_path  --output_dir
```
---

### Noise Maker
This Python script is designed for audio processing. It uses various libraries like `os`, `glob`, `librosa`, `numpy`, and `soundfile` to handle audio files. The core functionality is encapsulated in the `AudioProcessor` class, which processes audio files according to specified parameters. Here's a summary of its key features and functionalities:

1. **Initialization of Audio Processor**: The `AudioProcessor` class is initialized with several parameters:
   - `input_dirs`: A list of directories containing the input audio files.
   - `output_base_dir`: The base directory where processed audio files will be saved.
   - `input_length`: The desired length of the input audio in milliseconds.
   - `sample_rate`: The sample rate for audio processing.
   - `noise_factor`: A factor used to generate and mix random noise into the audio.
   - `silence_factor`: A factor to silence the audio.

2. **Output Path Determination**: `get_output_path` method determines the output path for processed files based on the input file's path.

3. **Processing Single Audio File**: The `process_single_audio` method performs several steps:
   - Loads an audio file using `librosa`.
   - Crops the audio to the desired length, centered around the midpoint of the audio.
   - Generates random noise and mixes it with the audio using the `noise_factor`.
   - Applies a `silence_factor` to the audio.
   - Writes the processed audio to the determined output path.

4. **Batch Processing**: The `process_audio` method allows batch processing of audio files. It:
   - Finds all `.wav` files in the specified input directories.
   - Uses multiprocessing (via `Pool`) to process multiple audio files in parallel for efficiency.

5. **Command-Line Interface**: The script is designed to be run from the command line, taking arguments for input directories, output base directory, input length, sample rate, silence factor, and noise factor.

6. **Error Handling**: The script includes basic error handling during audio processing, catching exceptions and printing error messages.

7. **Parallel Processing**: The use of the `multiprocessing.Pool` class allows for parallel processing of multiple audio files, enhancing performance on multi-core systems.

Overall, the script is a tool for batch processing of audio files, adding noise, and adjusting length and silence in the audio, with output saved in a specified location.

```bash
python make_noise.py --input_dirs [/path/to/input] --output_base_dir [/path/to/output] --input_length [3000] --sample_rate [16000] --silence_factor [0.01] --noise_factor [0.0007]
```