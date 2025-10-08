#!/usr/bin/python3

import sys
import getopt
import os
import pandas as pd
import numpy as np
from explainable_bert_classifier.data import fit_transform_LabelEncoder
from explainable_bert_classifier.data import tokenizer
from explainable_bert_classifier.data import CVEDataset
from explainable_bert_classifier.model import BertClassifier
from explainable_bert_classifier.model import early_stopping
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Use PyTorch's AdamW optimizer
import warnings
import time
import torch
from tqdm import tqdm  # For showing a progress bar

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning)

"""
Script to train CVSS-BERT classifiers
"""


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """
    Save model and optimizer checkpoint.
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load checkpoint to resume training.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
    return epoch, loss


def main(argv):
    input_data = ''
    output_dir = 'bert-classifier'
    metric_name = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:m:", ["input_data=", "output_dir=", "metric_name="])
    except getopt.GetoptError:
        print('train.py -i <input_data> -o <output_dir> -m <metric_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('train.py -i <input_data> -o <output_dir> -m <metric_name>')
            sys.exit()
        elif opt in ("-i", "--input_data"):
            input_data = arg
        elif opt in ("-o", "--output_dir"):
            output_dir = arg
        elif opt in ("-m", "--metric_name"):
            metric_name = arg

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + '/' + metric_name):
        os.makedirs(output_dir + '/' + metric_name)

    X_train = pd.read_csv(input_data + '_X_train.csv')
    y_train = pd.read_csv(input_data + '_y_train.csv')
    train_labels = y_train.loc[:, metric_name]
    encoded_train_labels = fit_transform_LabelEncoder(train_labels, save=True,
                                                      filename=output_dir + '/' + metric_name + '/label.txt')

    mytokenizer = tokenizer()
    train_encodings = mytokenizer(X_train.loc[:, "Description"].tolist(), truncation=True, padding=True, max_length=128)
    train_dataset = CVEDataset(X_train, train_encodings, train_labels, encoded_train_labels)

    print('Loading model...')
    NUM_CLASSES = len(set(train_labels))
    classifier = BertClassifier(num_labels=NUM_CLASSES)

    # Ensure classifier is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.model.to(device)

    # Define optimizer
    optimizer = AdamW(classifier.model.parameters(), lr=5e-5)

    # Load checkpoint if exists
    checkpoint_dir = os.path.join(output_dir, metric_name)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_latest.pth')

    if os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(checkpoint_path, classifier.model, optimizer)
        classifier.model.to(device)  # Ensure model is on the correct device
    else:
        print("No checkpoint found, starting training from scratch...")
        start_epoch = 0

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    print('Freeze base model')
    classifier.freeze_base(verbose=False)

    # Training with frozen base model, show progress bar
    print('Training with frozen base model...')
    classifier.model.train()
    total_batches = len(train_loader)
    for epoch in range(start_epoch, 2):  # Training for 2 epochs with frozen base
        print(f"Epoch {epoch + 1}/2 - Training with frozen base model...")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{2}", total=total_batches, ncols=100)

        training_loss = 0
        num_correct = 0
        num_examples = 0

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['encoded_labels'].to(device).long()

            outputs = classifier.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()

            training_loss += loss.data.item() * input_ids.size(0)
            correct = torch.eq(torch.max(torch.softmax(outputs.logits, dim=1), dim=1)[1], labels)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

            progress_bar.set_postfix(
                loss=training_loss / num_examples,
                accuracy=num_correct / num_examples,
            )

        # Save checkpoint after each epoch during "frozen base" phase
        save_checkpoint(classifier.model, optimizer, epoch + 1, training_loss, checkpoint_dir)

    print('Unfreeze base model')
    classifier.unfreeze_base(verbose=False)

    # Determine the optimal number of epochs
    print('Determining optimal number of training epochs using early stopping...')
    optimal_nb_epoch, history_early_stopping = early_stopping(classifier, X_train, train_labels, encoded_train_labels,
                                                              mytokenizer, max_epoch=8)
    print('Optimal number of training epoch: ', optimal_nb_epoch)

    # Training for optimal epochs with progress polling
    print('Training for optimal epochs with progress polling...')
    start_time = time.time()
    total_iterations = optimal_nb_epoch

    for epoch in range(start_epoch, total_iterations):
        print(f"Starting Epoch {epoch + 1}/{total_iterations}...")

        # Training for one epoch
        classifier.fit(loader=train_loader, total_iterations=1)

        # Save checkpoint at the end of each epoch
        save_checkpoint(classifier.model, optimizer, epoch + 1, None, checkpoint_dir)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 5:
            progress = (epoch + 1) / total_iterations * 100
            print(
                f"Progress: {progress:.2f}% - Epoch {epoch + 1}/{total_iterations} - Elapsed Time: {elapsed_time:.2f} seconds")
            sys.stdout.flush()  # Force the output to flush immediately
            start_time = time.time()  # Reset the timer after printing progress

    print('Saving model...')
    classifier.model.save_pretrained(output_dir + '/' + metric_name + '/model')
    print('Training completed. Model saved.')


if __name__ == "__main__":
    main(sys.argv[1:])
