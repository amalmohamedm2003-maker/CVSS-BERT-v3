import os
import torch
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from explainable_bert_classifier.data import CVEDataset, fit_transform_LabelEncoder
from tqdm import tqdm  # Progress bar library


def load_model_and_tokenizer(metric):
    """
    Load a pretrained model and its tokenizer for the given metric.
    Always use Hugging Face default tokenizer ('bert-base-uncased').
    """
    model_path = f"C:\\Users\\sayan\\Desktop\\cvss2223\\output_dir\\metric_name\\{metric}\\model"
    print(f"Loading model from {model_path}...")
    model = BertForSequenceClassification.from_pretrained(model_path)

    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer


def load_label_encoder(metric, label_data):
    """
    Load the label encoder for the given metric and fit it on the label data.
    """
    label_path = f"C:\\Users\\sayan\\Desktop\\cvss2223\\output_dir\\metric_name\\{metric}\\label.txt"
    print(f"Loading label encoder from {label_path}...")

    # Create and fit the label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(label_data)

    return label_encoder


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset and return predictions and ground truth labels.
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Add a progress bar for the evaluation
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", ncols=100, leave=False):  # tqdm added for progress bar
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['encoded_labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def main(input_data, metric):
    """
    Main function to load the model, tokenizer, test dataset, and evaluate metrics.
    """
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(metric)

    # Load test data
    print("Loading test data...")
    X_test = pd.read_csv(f"{input_data}_X_test.csv")
    y_test = pd.read_csv(f"{input_data}_y_test.csv")

    # Check if the metric is in the dataframe
    if metric not in y_test.columns:
        print(f"Error: Metric '{metric}' not found in y_test.")
        return

    test_labels = y_test[metric]

    # Load label encoder and fit it on the test labels
    label_encoder = load_label_encoder(metric, test_labels)

    # Apply the encoder to the test labels
    encoded_test_labels = label_encoder.transform(test_labels)

    # Tokenize test data
    print("Tokenizing test data...")
    test_encodings = tokenizer(X_test["Description"].tolist(), truncation=True, padding=True, max_length=128)
    test_dataset = CVEDataset(X_test, test_encodings, test_labels, encoded_test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Evaluate model
    print("Evaluating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, labels = evaluate_model(model, test_loader, device)

    # Compute metrics
    f1 = f1_score(labels, preds, average="weighted")
    accuracy = accuracy_score(labels, preds)
    balanced_acc = balanced_accuracy_score(labels, preds)

    print(f"Metrics for {metric}:")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a BERT model on test data.")
    parser.add_argument("--input_data", type=str, required=True,
                        help="Path to test data without suffix (_X_test/_y_test).")
    parser.add_argument("--metric", type=str, required=True, help="Name of the metric being evaluated.")

    args = parser.parse_args()
    main(args.input_data, args.metric)
