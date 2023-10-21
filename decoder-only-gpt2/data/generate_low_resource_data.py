"""This script samples K examples randomly without replacement from the original data."""

import argparse
import csv
import os
import numpy as np
import pandas as pd
import shutil
from pandas import DataFrame

def get_label(task, line):
    if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "sst-2", "CoLA"]:
        # GLUE style
        # line = line.strip().split('\t')
        if task == 'CoLA':
            return line[1]
        elif task == 'MNLI':
            return line[-1]
        elif task == 'MRPC':
            return line[0]
        elif task == 'QNLI':
            return line[-1]
        elif task == 'QQP':
            return line[-1]
        elif task == 'RTE':
            return line[-1]
        elif task == 'SNLI':
            return line[-1]
        elif task == 'sst-2':
            return line[-1]
        else:
            raise NotImplementedError
    else:
        return line[0]

def load_datasets(data_dir, tasks):
    datasets = {}
    for task in tasks:
        dataset = {}
        dirname = os.path.join(data_dir, task)
        if task == "MNLI":
            splits = ["train", "dev_matched", "dev_mismatched"]
        elif task in ['mpqa', 'mr', 'subj', 'trec']:
            splits = ["train"]
        else:
            splits = ["train", "dev"]
        for split in splits:
            filename = os.path.join(dirname, f"{split}.tsv")
            with open(filename, "r") as f:
                lines = list(csv.reader(f, delimiter="\t", quotechar=None))
            dataset[split] = lines
        datasets[task] = dataset
    return datasets

def split_header(task, lines):
    """
    Returns if the task file has a header or not. Only for GLUE tasks.
    """
    if task in ["cola", "subj", "mpqa", 'trec', 'mr']:
        return [], lines
    elif task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "sst-2"]:
        return lines[0:1], lines[1:]
    else:
        raise ValueError("Unknown task.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=500,
        help="Training examples for each class.")
    parser.add_argument("--k_for_dev", type=int, default=1000,
        help="Dev examples for each class.")
    parser.add_argument("--task", type=str, nargs="+", 
        default=["MNLI", "QQP", "RTE", "QNLI", "MRPC", "SNLI"],
        help="Task names")
    parser.add_argument("--seed", type=int, nargs="+", 
        default=[100, 13, 21, 42, 87],
        help="Random seeds")

    parser.add_argument("--data_dir", type=str, default="../datasets/glue_data", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="../datasets/late-prompt", help="Output path")
    parser.add_argument("--mode", type=str, default='k-shot-10x', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x dev set)") 

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, '{}-samples'.format(args.k))

    k = args.k
    print("K =", k)
    datasets = load_datasets(args.data_dir, args.task)

    for seed in args.seed:
        print("Seed = %d" % (seed))
        output_dir = os.path.join(args.output_dir, 'seed-{}'.format(seed))
        for task, dataset in datasets.items():
            # Set random seed
            np.random.seed(seed)

            # Shuffle the training set
            print("| Task = %s" % (task))

            train_header, train_lines = split_header(task, dataset["train"])
            np.random.shuffle(train_lines)

            # Set up dir
            task_dir = os.path.join(output_dir, task.lower())
            # setting_dir = os.path.join(task_dir, f"{k}-{seed}")
            os.makedirs(task_dir, exist_ok=True)

            # Write test splits
            if task in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "sst-2", "cola"]:
                # GLUE style
                # Use the original development set as the test set (the original test sets are not publicly available)
                for split, lines in dataset.items():
                    if split.startswith("train"):
                        continue
                    split = split.replace('dev', 'test')
                    with open(os.path.join(task_dir, f"{split}.tsv"), "w") as f:
                        writer = csv.writer(f, delimiter="\t", quotechar=None)
                        for line in lines:
                            writer.writerow(line)

            label_list = {}
            for line in train_lines:
                label = get_label(task, line)
                if label not in label_list:
                    label_list[label] = [line]
                else:
                    label_list[label].append(line)

            num_per_class = []
            for label, data_per_class in label_list.items():
                num_per_class.append(len(data_per_class))
            
            total_num = sum(num_per_class)
            selected_num_per_class = [round(args.k * i / total_num) for i in num_per_class[0:-1]]
            selected_num_per_class.append(args.k - sum(selected_num_per_class))

            assert len(list(label_list.keys())) == len(selected_num_per_class)

            with open(os.path.join(task_dir, "train.tsv"), "w") as f:
                writer = csv.writer(f, delimiter="\t", quotechar=None)
                for line in train_header:
                    writer.writerow(line)
                for label,  k_per_class in zip(label_list, selected_num_per_class):
                    if len(label_list[label]) < k_per_class:
                        for line in label_list[label]:
                            writer.writerow(line)
                    else:
                        for line in label_list[label][:k_per_class]:
                            writer.writerow(line)

            selected_num_per_class_dev = [round(args.k_for_dev * (n - selected_num_per_class[i])/ (total_num - args.k)) for i, n in enumerate(num_per_class[0:-1])]
            selected_num_per_class_dev.append(args.k_for_dev - sum(selected_num_per_class_dev))            
            names = ["dev.tsv"]
            if task == 'MNLI':
                names = ["dev_matched.tsv", "dev_mismatched.tsv"]
            for name in names:
                with open(os.path.join(task_dir, name), "w") as f:
                    writer = csv.writer(f, delimiter="\t", quotechar=None)
                    for line in train_header:
                        writer.writerow(line)
                    for label, k_per_class_dev, k_per_class in zip(label_list, selected_num_per_class_dev, selected_num_per_class):
                        if len(label_list[label]) < (k_per_class + k_per_class_dev):
                            for line in label_list[label][k_per_class:]:
                                writer.writerow(line)
                        else:                          
                            for line in label_list[label][k_per_class:(k_per_class + k_per_class_dev)]:
                                writer.writerow(line)
            
            if task in ["subj", "mpqa", 'trec', 'mr']:
                src_file = os.path.join(args.data_dir, task, 'test.tsv')
                shutil.copy(src_file, task_dir)


if __name__ == "__main__":
    main()