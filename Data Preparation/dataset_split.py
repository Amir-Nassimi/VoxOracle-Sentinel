import os
import random
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def split_dataset(dataset_dir, train_dir, valid_dir, test_dir, train_csv_path, valid_csv_path, test_csv_path, valid_split=0.1, test_split=0.1):
    """
    Split the dataset into training, validation, and testing sets and save the split information in CSV files.
    """
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(valid_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)

    train_data = []
    valid_data = []
    test_data = []

    for label_dir in tqdm(os.listdir(dataset_dir), desc="Processing labels"):
        label_path = os.path.join(dataset_dir, label_dir)

        if os.path.isdir(label_path):
            files = [file for file in os.listdir(label_path) if file.endswith('.npy')]
            random.shuffle(files)

            num_valid_samples = int(len(files) * valid_split)
            num_test_samples = int(len(files) * test_split)
            num_train_samples = len(files) - num_valid_samples - num_test_samples

            train_files = files[:num_train_samples]
            valid_files = files[num_train_samples:num_train_samples + num_valid_samples]
            test_files = files[num_train_samples + num_valid_samples:]

            train_label_dir = Path(train_dir) / label_dir
            valid_label_dir = Path(valid_dir) / label_dir
            test_label_dir = Path(test_dir) / label_dir
            train_label_dir.mkdir(parents=True, exist_ok=True)
            valid_label_dir.mkdir(parents=True, exist_ok=True)
            test_label_dir.mkdir(parents=True, exist_ok=True)

            for file in tqdm(train_files, desc=f"Copying train files for {label_dir}"):
                src_path = os.path.join(label_path, file)
                dest_path = os.path.join(train_label_dir, file)
                shutil.copy2(src_path, dest_path)
                train_data.append({'file_path': dest_path, 'label': label_dir})

            for file in tqdm(valid_files, desc=f"Copying valid files for {label_dir}"):
                src_path = os.path.join(label_path, file)
                dest_path = os.path.join(valid_label_dir, file)
                shutil.copy2(src_path, dest_path)
                valid_data.append({'file_path': dest_path, 'label': label_dir})

            for file in tqdm(test_files, desc=f"Copying test files for {label_dir}"):
                src_path = os.path.join(label_path, file)
                dest_path = os.path.join(test_label_dir, file)
                shutil.copy2(src_path, dest_path)
                test_data.append({'file_path': dest_path, 'label': label_dir})

    pd.DataFrame(train_data).to_csv(f'{train_csv_path}/Train.csv', index=False)
    pd.DataFrame(valid_data).to_csv(f'{valid_csv_path}/Valid.csv', index=False)
    pd.DataFrame(test_data).to_csv(f'{test_csv_path}/Test.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into training, validation, and testing sets and create CSV files for the split.")

    parser.add_argument('--train_dir', type=str, required=True, help='Path to save training set')
    parser.add_argument('--valid_dir', type=str, required=True, help='Path to save validation set')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to save testing set')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to save the training set CSV file')
    parser.add_argument('--valid_csv', type=str, required=True, help='Path to save the validation set CSV file')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to save the testing set CSV file')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--valid_split', type=float, required=False, default=0.1, help='Fraction of data to be used as validation set')
    parser.add_argument('--test_split', type=float, required=False, default=0.1, help='Fraction of data to be used as testing set')

    args = parser.parse_args()

    split_dataset(args.dataset_dir, args.train_dir, args.valid_dir, args.test_dir, args.train_csv, args.valid_csv,
                  args.test_csv, args.valid_split, args.test_split)


if __name__ == "__main__":
    main()
