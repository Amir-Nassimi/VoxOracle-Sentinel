import os
import argparse
import random
import shutil
import pandas as pd
from pathlib import Path


def split_dataset(dataset_dir, train_dir, valid_dir, train_csv_path, valid_csv_path, valid_split=0.2):
    """
    Split the dataset into training and validation sets and save the split information in CSV files.
    """
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)

    train_data = []
    valid_data = []

    # Iterate over each label's subdirectory
    for label_dir in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label_dir)

        if os.path.isdir(label_path):
            files = [file for file in os.listdir(label_path) if file.endswith('.npy')]
            random.shuffle(files)

            # Split dataset
            num_valid_samples = int(len(files) * valid_split)
            train_files = files[num_valid_samples:]
            valid_files = files[:num_valid_samples]

            # Create label subdirectories in train and valid directories
            train_label_dir = Path(train_dir) / label_dir
            valid_label_dir = Path(valid_dir) / label_dir
            train_label_dir.mkdir(parents=True, exist_ok=True)
            valid_label_dir.mkdir(parents=True, exist_ok=True)

            # Copy files to respective directories and append to lists
            for file in train_files:
                src_path = os.path.join(label_path, file)
                dest_path = os.path.join(train_label_dir, file)
                shutil.copy2(src_path, dest_path)
                train_data.append({'file_path': dest_path, 'label': label_dir})

            for file in valid_files:
                src_path = os.path.join(label_path, file)
                dest_path = os.path.join(valid_label_dir, file)
                shutil.copy2(src_path, dest_path)
                valid_data.append({'file_path': dest_path, 'label': label_dir})

    # Save file paths to CSV files
    pd.DataFrame(train_data).to_csv(f'{train_csv_path}/Train.csv', index=False)
    pd.DataFrame(valid_data).to_csv(f'{valid_csv_path}/Validation.csv', index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Split a dataset into training and validation sets and create CSV files for the split.")

    parser.add_argument('--train_dir', type=str, required=True, help='Path to save training set')
    parser.add_argument('--valid_dir', type=str, required=True, help='Path to save validation set')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to save the training set CSV file')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path to save the validation set CSV file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--valid_split', type=float, required=False, default=0.2,
                        help='Fraction of data to be used as validation set (default: 0.2)')

    args = parser.parse_args()

    split_dataset(args.dataset_dir, args.train_dir, args.valid_dir, args.train_csv, args.valid_csv, args.valid_split)


if __name__ == "__main__":
    main()
