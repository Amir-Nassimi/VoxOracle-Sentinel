''' Offline inference from hotword detection model '''
import os
import argparse
import sys
from pathlib import Path

sys.path.append(os.path.abspath(Path(__file__).resolve().parents[0]))
from transcribe import Transcribe


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, required=True, help="model path")
    parser.add_argument("--input", type=str, required=True, help="input audio file")
    parser.add_argument("--sr", type=int, required=False, default=16000, help="model sample rate")
    parser.add_argument("--step", type=float, required=False, default=0.4, help="The step size or stride of the sliding window. It determines how much the window moves for each subsequent analysis. A smaller step size means more overlap between consecutive windows and can lead to finer resolution in time, but at the cost of increased computational load. A step size of 0.4 seconds is a balance that allows for detailed analysis without excessive computation.")
    parser.add_argument("--window_size", type=float, required=False, default=1.9, help="The size of the window used for analysis, measured in seconds. A window of 1.9 seconds will be used to slice the audio signal into segments for processing. The choice of window size is critical; a larger window can capture more temporal context but might mix different sound events, while a smaller window may focus on instantaneous features but at the risk of losing context.")

    args = parser.parse_args()

    labels = {0: 'Ahange-Baadi', 1: 'Ahange-Ghabli', 2: 'Barname-Aval', 3: 'Barname-Chaharom',
              4: 'Barname-Dahom', 5: 'Barname-Dovom', 6: 'Barname-Panjom'}

    transcribe = Transcribe(args.model_pth, labels, args.step, args.window_size)
    result = transcribe.offline_inference(args.input)
    print(result)


if __name__ == "__main__":
    main()
