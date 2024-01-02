import argparse
from transcribe import Transcribe


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_pth", type=str, required=True, help="model path")

    args = parser.parse_args()

    labels = {0: 'Ahange-Baadi', 1: 'Ahange-Ghabli', 2: 'Barname-Aval', 3: 'Barname-Chaharom',
              4: 'Barname-Dahom', 5: 'Barname-Dovom', 6: 'Barname-Panjom'}

    transcribe = Transcribe(args.model_pth, labels)
    result = transcribe.online_inference()
    print(result)


if __name__ == "__main__":
    main()
