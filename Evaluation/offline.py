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

    args = parser.parse_args()

    labels = {0: 'Ahange-Baadi', 1: 'Ahange-Ghabli', 2: 'Barname-Aval', 3: 'Barname-Chaharom',
              4: 'Barname-Dahom', 5: 'Barname-Dovom', 6: 'Barname-Panjom'}

    transcribe = Transcribe(args.model_pth, labels)
    result = transcribe.offline_inference(args.input)
    print(result)


if __name__ == "__main__":
    main()
