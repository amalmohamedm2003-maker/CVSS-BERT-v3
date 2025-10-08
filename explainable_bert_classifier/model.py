import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
import copy
from explainable_bert_classifier.data import split_dataset


class BertClassifier:
    """
    BERT classifier object.
    """

    def __init__(self, model_name="prajjwal1/bert-small", **kwargs):
        """
        Initialize a BERT model for sequence classification tasks.
        """
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_name, **kwargs)

    def freeze_base(self, verbose=True):
        """
        Freeze the base BERT model.
        """
        for param in self.model.base_model.parameters():
            param.requires_grad = False
        if verbose:
            self.print_trainable_parameters()

    def unfreeze_base(self, verbose=True):
        """
        Unfreeze the base BERT model.
        """
        for param in self.model.base_model.parameters():
            param.requires_grad = True
        if verbose:
            self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """
        Print the total number of parameters and how many of them are trainable.
        """
        print(
            "Total number of trainable parameters:",
            sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        )
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Trainable parameters:", name, param.data)

    def fit(self, loader, total_iterations, optim=AdamW, lr=5e-5):

        optim = optim(self.model.parameters(), lr=lr)
        self.model.to(self.device)
        self.model.train()

        training_loss_epoch = []
        training_loss_batch = []
        training_accuracy_epoch = []

        for epoch in range(total_iterations):
            training_loss = 0
            num_correct = 0
            num_examples = 0

            print(f"Epoch {epoch + 1}/{total_iterations}")
            batch_iterator = tqdm(loader, desc=f"Training Epoch {epoch + 1}", unit="batch")

            for batch in batch_iterator:
                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["encoded_labels"].to(self.device)

                # Ensure labels are of type Long for compatibility with loss
                labels = labels.long()

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                loss.backward()
                optim.step()

                training_loss_batch.append(loss.item())
                training_loss += loss.item() * input_ids.size(0)

                predictions = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
                num_correct += torch.sum(predictions == labels).item()
                num_examples += labels.size(0)

                # Update progress bar description
                batch_iterator.set_postfix(loss=loss.item(), accuracy=num_correct / num_examples)

            epoch_loss = training_loss / len(loader.dataset)
            epoch_accuracy = num_correct / num_examples
            training_loss_epoch.append(epoch_loss)
            training_accuracy_epoch.append(epoch_accuracy)

            print(f"Epoch {epoch + 1} Complete: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")

        return {
            "training_loss_per_epoch": training_loss_epoch,
            "training_loss_per_batch": training_loss_batch,
            "training_accuracy_per_epoch": training_accuracy_epoch,
        }

    def evaluate_batch_by_batch(self, loader):
        """
        Evaluate data batch by batch (to avoid loading all the samples in memory when the dataset is large). Compute loss and accuracy.
        """
        self.model.to(self.device)
        self.model.eval()
        num_correct = 0
        num_examples = 0
        test_loss = 0
        predicted_labels_list = []
        predicted_labels_score_list = []

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["encoded_labels"].to(self.device)

            # Convert labels to Long type
            labels = labels.long()

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            test_loss += loss.data.item() * input_ids.size(0)
            predicted_labels = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
            predicted_labels_list.extend(predicted_labels.tolist())
            predicted_labels_score = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[0]
            predicted_labels_score_list.extend(predicted_labels_score.tolist())
            correct = torch.eq(predicted_labels, labels)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]

        test_loss /= len(loader.dataset)
        accuracy = num_correct / num_examples

        print(f"Loss: {test_loss}, Accuracy = {accuracy}")
        return {
            "predicted_labels": predicted_labels_list,
            "predicted_scores": predicted_labels_score_list,
            "accuracy": accuracy,
            "loss": test_loss,
        }

    def predict(self, batch_tokenized):
        """
        Predict the labels for a batch of samples.
        """
        self.model.to(self.device)
        self.model.eval()

        input_ids = torch.tensor(batch_tokenized["input_ids"]).to(self.device)
        attention_mask = torch.tensor(batch_tokenized["attention_mask"]).to(self.device)

        if len(input_ids.size()) == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        predicted_labels = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1]
        predicted_labels_score = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[0]

        return {"predicted_labels": predicted_labels, "predicted_scores": predicted_labels_score}


def early_stopping(
    classifier, dataset, labels, encoded_labels, tokenizer, val_proportion=0.2, max_epoch=10, optim=AdamW, lr=5e-5
):
    """
    Compute the optimal number of epochs using early stopping. A copy of the classifier is trained.
    """
    classifier_copy = BertClassifier(num_labels=len(set(labels)))
    classifier_copy.model = copy.deepcopy(classifier.model)
    analysis_dataset, assessment_dataset = split_dataset(
        dataset, labels, encoded_labels, tokenizer, val_proportion, shuffle=True
    )

    train_loader = DataLoader(analysis_dataset, batch_size=16)
    val_loader = DataLoader(assessment_dataset, batch_size=16)

    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(max_epoch):
        print(f"Epoch {epoch + 1}/{max_epoch}")
        print("Training...")
        train_eval = classifier_copy.fit(loader=train_loader, total_iterations=1, optim=optim, lr=lr)
        train_loss = train_eval["training_loss_per_epoch"][0]
        train_acc = train_eval["training_accuracy_per_epoch"][0]

        print("Validation...")
        val_eval = classifier_copy.evaluate_batch_by_batch(val_loader)
        val_loss = val_eval["loss"]
        val_acc = val_eval["accuracy"]

        # Save metrics
        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= 2:
            print("Validation loss did not improve for 2 consecutive epochs. Stopping early.")
            break

        print(
            f"Epoch {epoch + 1} Summary: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"
        )

    optimal_nb_epoch = len(history["val_loss"])
    print(f"Optimal number of epochs: {optimal_nb_epoch}")
    return optimal_nb_epoch, history

