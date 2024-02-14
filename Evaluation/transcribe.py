import os
import sys
import librosa
import numpy as np
from pathlib import Path
from pydub import AudioSegment
from collections import Counter

from streamer import Streamer
from frame_proc import FrameASR

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[1]))
from Train.dense_net import ModelBuilder


class Transcribe:
    def __init__(self, model_path, lbl_source, step, window_size, attention_type, softmax_type, sparsity_rate,
                 sample_rate=16000, history_len=6, in_shape=(128, 211, 3), pr_acc=30):

        self.sample_rate = sample_rate
        self.history_len = history_len
        self.chunk_size = int(step * sample_rate)

        self.model = ModelBuilder(in_shape).build_model(len(lbl_source), attention_type, softmax_type, sparsity_rate)
        try:
            self.model.load_weights(model_path)
            print("Weights loaded successfully!!")
        except Exception as error:
            print(f"ValueError: Error on loading weights: {error}")
            raise ValueError(f"Error on loading weights: {error}")

        self.mbn = FrameASR(model=self.model, frame_len=step, target_len=window_size,
                            frame_overlap=((window_size-step)/2), label_source=lbl_source, pr_acc=pr_acc)

    @staticmethod
    def generate_audio_chunks(audio, chunk_size):
        # Generator to yield chunks of audio
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]

    def offline_inference(self, wave_file):
        mbn_history, detection_info = [], []

        audio, _ = librosa.load(wave_file, sr=self.sample_rate, mono=True)

        # Iterate over chunks using the generator
        for chunk in self.generate_audio_chunks(audio, self.chunk_size):
            # Process the chunk
            chunk = np.pad(chunk, (0, max(0, self.chunk_size - len(chunk))), 'constant')
            signal, mbn_result = self._process_signal(chunk)
            mbn_history.append(mbn_result)

            # Handle history and most common command calculation
            if len(mbn_history) > self.history_len:
                mbn_history.pop(0)

        # Calculate the most common command after processing all chunks
        print(mbn_history)
        most_common_cmd = self._get_most_common_cmd(mbn_history)
        return most_common_cmd

    @staticmethod
    def _get_most_common_cmd(mbn_history):
        counter = Counter(mbn_history)
        return counter.most_common(1)[0]

    @staticmethod
    def _overlay_chime(chime_file, wave_name, wave_file, outdir, detection_time):
        audio_clip = AudioSegment.from_wav(wave_file)
        chime = AudioSegment.from_wav(chime_file)
        audio_clip = audio_clip.overlay(chime, position=int(detection_time.split(':')[2])*1000)
        audio_clip.export(f"{outdir}/{os.path.splitext(wave_name)[0]}_{detection_time}_output.wav", format='wav')

    def _process_signal(self, data):
        signal = np.frombuffer(data, dtype=np.int16).astype(np.float32) / (2**15)

        # Convert stereo to mono by averaging both channels
        if data.shape != signal.shape:
            signal = (signal[0::2] + signal[1::2]) / 2

        return signal, self.mbn.transcribe(signal)

    def online_inference(self):
        stream = Streamer(self.chunk_size).stream
        mbn_history = []

        self.mbn.reset()

        while True:
            data = stream.read(self.chunk_size)
            signal = np.frombuffer(data, dtype=np.int16)
            signal = signal.astype(np.float32)/32768.
            mbn_result, data = self.mbn.transcribe(signal)
            mbn_history[:-1] = mbn_history[1:]

            if mbn_result==['heyholoo']:
                mbn_history[-1]=1
            else:
                mbn_history[-1]=0

            if mbn_history[1:-1]==[0,0] and mbn_history[-1]==1:
                result = {'command':'heyholoo', 'detect_time': datetime.datetime.now().strftime('%H:%M:%S')}
                print(result)

        print('Listening...')
        stream.start_stream()

        # Interrupt kernel and then speak for a few more words to exit the pyaudio loop !
        try:
            while stream.is_active():
                sleep(0.1)

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            print()
            print("PyAudio stopped")
