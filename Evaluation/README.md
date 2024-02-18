# VoxOracle Sentinel

## Evaluating The Model

### Online run (online.py)

**Purpose**

- Provide a command line interface (CLI) for real-time speech transcription 
- Allow passing parameters to configure Transcribe behavior

- **Key Components**:  

  - `parser` - To define and handle command line arguments

  - `main()`
    - Load model, labels, and config parameters from CLI args
    - Initialize Transcribe with those arguments
    - Call online_inference() to start real-time transcription

So in essence, it:

1. Handles command line interaction
2. Parses out key parameters  
3. Passes those params to instantiate the Transcribe workflow
4. Starts live transcription using mic input

This allows customizing parameters like:

- Model path
- Labels 
- Audio length hyperparameters 
- Model architecture choices
- Prediction confidence threshold

And then performs streaming speech transcription accordingly through the Transcribe pipeline. So it provides a configurable entry point harnessing Transcribe's underlying functionality.

**Caution** : Don't forget to add the pickle file containing your desired labels. To do so, we suggest utilizing pickle.dumb to create pickle file. The overall format should be similar to the following pattern:
  ```bash
  labels = {
      0: 'Label1',
      1: 'Label2',
      2: 'Label3',
      3: 'Label4',
      # ... and so on for other labels
  }
  ```

---

### Offline run (offline.py)

**Purpose**:

- Provide a command line tool to transcribe a given audio file using a pretrained model
- Allow customizing Transcribe parameters  

- **Key Components**:

  - `parser`:  
    - Defines arguments like model path, audio file, output labels etc
  
  - `main()`:
    - Load model, labels, config params from CLI args 
    - Create Transcribe object configured with those params
    - Call offline_inference() on given audio file path
    - Print out the transcription result
  
Overall it provides an easy way to:

1. Load model and labels
2. Parse command line for input file and parameters
3. Initialize a Transcribe instance 
4. Run that Transcribe workflow on the file 
5. Output transcribed text

This enables configuring Transcribe to user's needs, while handling the underlying model and feature extraction automatically.

Some key parameters exposed:
- Audio window size
- Step size between windows
- Model sparsity and attention settings
- Output label set customization

So in summary, it adds a simple harness for transcribing audio files leveraging Transcribe's capabilities.

**Caution** : Don't forget to add the pickle file containing your desired labels. To do so, we suggest utilizing pickle.dumb to create pickle file. The overall format should be similar to the following pattern:
  ```bash
  labels = {
      0: 'Label1',
      1: 'Label2',
      2: 'Label3',
      3: 'Label4',
      # ... and so on for other labels
  }
  ```

---

### Transcribe (transcribe.py)

**Overview**

The Transcribe class handles an end-to-end audio transcription pipeline, from input handling to providing transcribed text output. It supports both offline and live microphone audio as input.

It leverages several helper classes for core functionality:

1. Streamer: Handles getting live audio data from the microphone using sounddevice callbacks and putting it into a buffer/queue.

2. FrameASR: Handles audio frame preprocessing, feeding frames into a TensorFlow model, and postprocessing the output. Designed for low latency transcription.

3. ModelBuilder: Assembles the deep learning model architecture and provides access to load pretrained weights.

The Transcribe class brings these together into an overall workflow.

- Key Functions:

  - `__init__()`:
    - Loads the pretrained weights into the model 
    - Sets up the FrameASR and Streamer helpers
    - Defines runtime parameters like audio buffer length
  
  - `generate_audio_chunks()`:
    - Helper generator to yield fixed size chunks of audio for processing
  
  - `offline_inference()`:
    - Entry point for transcribing an audio file
    - Loads the full audio, splits it into chunks
    - Feeds each chunk to FrameASR in sequence
    - Aggregates the results across all chunks
  
  - `online_inference()`:
    - Entry point for live microphone transcription 
    - Starts the Streamer callback
    - Gets audio chunks from the Streamer continuously 
    - Feeds each chunk to FrameASR
    - Aggregates the incremental results
  
  - `_process_signal()`:
    - Helper method to handle audio conversion
    - Passes the chunk to FrameASR
    - Gets back the transcription output

_**So in summary:**_

- Handles two pathways for offline files and live audio 
- Manages buffering, chunking, aggregation
- Leaves model and audio handling to other classes
- Orchestrates the overall pipeline

---

### Frame Processing (frame_proc.py)

The FrameASR class is designed to handle audio frame processing for streaming speech recognition. It takes incoming audio frames, handles buffering and feature extraction, runs model inference, and decodes the output.

The purpose of FrameASR is to enable low latency transcription by operating on small chunks of audio input. But many speech models require larger context, so it accumulates the audio frames into an internal rolling buffer.

- **Main Attributes**:
  - `model` - The TensorFlow neural network model for inference
  - `frame_len` - Single input frame length in seconds 
  - `target_len` - Total spectrogram length expected by the model 
  - `sample_rate` - Audio sample rate
  - `label_source` - Dictionary mapping integer indices to text
  - `buffer` - Ring buffer to accumulate audio frame history

- **Key Methods**:  

  - `__init__()` - Initialize parameters and rolling buffer
  - `reset()` - Clear and re-populate the buffer with zeros
  - `transcribe()` - Main entry point
    - Add new frame to buffer
    - Extract mel spectrogram features
    - Run model inference 
    - Decode output text prediction
  - `decode_pred()` - Postprocess output
    - Apply confidence threshold
    - Convert index to text label

So in essence, FrameASR handles the streaming buffer, feature extraction, model integration, and postprocessing - everything needed for low latency speech transcription.

---

### Audio Streamer (streamer.py)

**Purpose**

The Streamer class handles getting live audio input from a microphone or other input device using the sounddevice library. It provides a simple interface to read audio frames suitable for speech recognition or other streaming applications.

**Key Features**
- Finds available audio devices 
- Automatically selects default device if none specified
- Starts low latency audio streaming in a background thread
- Provides stream property to access and read input frames
- Designed as a singleton to enable shared access

**Methods**

- `__init__`Initialize parameters
  - Checks for devices, allows selection
  - Starts InputStream in background
  
- `get_available_devices` - Helper to list available mics/devices

- `choose_device` - Lets user pick device by index 

- `stream` - Property provides access to sounddevice InputStream
  - Reads support blocking/callback/non-blocking modes
  
So in summary, it handles:

- Device selection
- Starting background streaming 
- Exposing microphone data through stream property

Allowing other classes to easily get live audio input without needing to handle low-level stream creation and querying devices.