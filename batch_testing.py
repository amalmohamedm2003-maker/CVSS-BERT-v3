import os
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from torch.utils.data import DataLoader
from transformers import BertTokenizer
from explainable_bert_classifier.data import CVEDataset
from sklearn.preprocessing import LabelEncoder
from testing import load_model_and_tokenizer, load_label_encoder, evaluate_model  # Reuse functions

def run_all_metrics(input_data):
    metrics = [
        "cvssV3_attackVector",
        "cvssV3_attackComplexity",
        "cvssV3_privilegesRequired",
        "cvssV3_userInteraction",
        "cvssV3_scope",
        "cvssV3_confidentialityImpact",
        "cvssV3_integrityImpact",
        "cvssV3_availabilityImpact"
    ]

    # Load raw test data once
    print("Loading test data...")
    X_test = pd.read_csv(f"{input_data}_X_test.csv")
    y_test = pd.read_csv(f"{input_data}_y_test.csv")

    # Tokenize once using any default tokenizer (e.g., from the first metric's model)
    sample_metric = metrics[0]
    _, tokenizer = load_model_and_tokenizer(sample_metric)
    print("Tokenizing test data (once)...")
    test_encodings = tokenizer(X_test["Description"].tolist(), truncation=True, padding=True, max_length=128)

    # Use same device across evaluations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for metric in metrics:
        print("\n" + "="*60)
        print(f"Evaluating model for metric: {metric}")
        print("="*60)

        if metric not in y_test.columns:
            print(f"Skipping {metric}: Column not found in y_test.")
            continue

        test_labels = y_test[metric]
        label_encoder = load_label_encoder(metric, test_labels)
        encoded_test_labels = label_encoder.transform(test_labels)

        test_dataset = CVEDataset(X_test, test_encodings, test_labels, encoded_test_labels)
        test_loader = DataLoader(test_dataset, batch_size=16)

        model, _ = load_model_and_tokenizer(metric)

        preds, labels = evaluate_model(model, test_loader, device)

        from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
        f1 = f1_score(labels, preds, average="weighted")
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)

        print(f"  F1 Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")


if __name__ == "__main__":
    input_data = r"C:\Users\sayan\Desktop\cvss2223\data\cve_2022-2023"
    run_all_metrics(input_data)
