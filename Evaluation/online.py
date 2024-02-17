import pickle
import argparse
from transcribe import Transcribe


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_pth", type=str, required=True, help="model path")
    parser.add_argument("--labels_file", type=str, required=True, help="path to labels pickle file")
    parser.add_argument("--acc_thrs", type=int, required=False, default=30,
                        help="Prediction acc threshold; default: 30%")
    parser.add_argument("--in_shape", type=str, required=False, default='128,211',
                        help="input audio shape the model is trained with; default: 128,211")
    parser.add_argument("--sr", type=int, required=False, default=16000,
                        help="input audio Sample Rate - if None, use the Mic Sample rate; default: 16000")
    parser.add_argument("--step", type=float, required=False, default=0.4,
                        help="The step size or stride of the sliding window. It determines how much the window moves for each subsequent analysis. A smaller step size means more overlap between consecutive windows and can lead to finer resolution in time, but at the cost of increased computational load. A step size of 0.4 seconds is a balance that allows for detailed analysis without excessive computation.")
    parser.add_argument("--window_size", type=float, required=False, default=2.1,
                        help="The size of the window used for analysis, measured in seconds. A window of 1.9 seconds will be used to slice the audio signal into segments for processing. The choice of window size is critical; a larger window can capture more temporal context but might mix different sound events, while a smaller window may focus on instantaneous features but at the risk of losing context.")
    parser.add_argument('--initial_sparsity_rate', type=float, required=False, default=0.5,
                        help='The Initial Sparsity Rate of R_Softmax; default value: 0.5')
    parser.add_argument('--softmax_type', type=str, required=False, default='normal',
                        help='The Softmax Strategy to use; Available Strategies: "normal" (Softmax), "r_softmax" (R_Softmax)')
    parser.add_argument('--attention_type', type=str, required=False, default='no_attention',
                        help='Whether to utilize an Attention Strategy or not; "normal" (Normal Attention Layer), "gaap" (GAAP Attention Layer), default: not to utilize any')

    args = parser.parse_args()

    # labels = {0:'Ahange-Ghabli',1:'Barname-Panjom',2:'Seda-Kam-Kon'}

    try:
        with open(args.labels_file, 'rb') as file:
            labels = pickle.load(file)
    except pickle.UnpicklingError as error:
        raise ValueError(f'The Pickle file is corrupted !! try creating it via pickle.dump - the error: {error}')

    in_shape = tuple(map(int, args.in_shape.split(','))) + (3,)

    transcribe = Transcribe(args.model_pth, labels, args.step, args.window_size, args.attention_type,
                            args.softmax_type, args.initial_sparsity_rate, in_shape=in_shape, pr_acc=args.acc_thrs,
                            sample_rate=args.sr)

    transcribe.online_inference()


if __name__ == "__main__":
    main()
