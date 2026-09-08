import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src import config
from src.dataset import CassavaDataset, get_transforms
from src.model import get_model


def build_sampler(labels):
    """Weighted sampler so under-represented classes get sampled more often."""
    label_unique, counts = np.unique(labels, return_counts=True)
    weights = [sum(counts) / c for c in counts]
    sample_weights = [weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_dataloaders():
    train_df = pd.read_csv(config.BASE_DIR + "train.csv")
    x_train, x_val, y_train, y_val = train_test_split(
        train_df["image_id"], train_df["label"],
        test_size=0.2, random_state=config.SEED, stratify=train_df["label"]
    )

    train_ds = CassavaDataset(x_train.values, y_train.values, get_transforms("train"))
    val_ds = CassavaDataset(x_val.values, y_val.values, get_transforms("val"))

    train_loader = DataLoader(
        train_ds, batch_size=config.TRAIN_BATCH_SIZE, num_workers=config.NUM_WORKERS,
        sampler=build_sampler(y_train.values)
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.VAL_BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False
    )
    return train_loader, val_loader


def train_model(model, train_loader, val_loader, num_epochs=15, patience=4):
    """
    Trains with early stopping and actually keeps the best-performing weights
    (the original notebook computed best_model_wts but never restored it).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start = time.time()
    for epoch in range(num_epochs):
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()

            running_loss, running_corrects, total = 0.0, 0, 0

            progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs} [{phase}]", leave=False)
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += inputs.size(0)

                progress_bar.set_postfix(loss=running_loss / total, acc=running_corrects / total)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"Epoch {epoch+1}/{num_epochs} [{phase}] loss: {epoch_loss:.4f} acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping early.")
            break

    elapsed = time.time() - start
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s. Best val acc: {best_acc:.4f}")

    model.load_state_dict(best_wts)
    return model, history


def plot_history(history, save_path="assets/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training curves to {save_path}")


def evaluate(model, val_loader, class_names, save_path="assets/confusion_matrix.png"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved confusion matrix to {save_path}")


if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")

    train_loader, val_loader = get_dataloaders()
    model = get_model(freeze_backbone=False)

    model, history = train_model(model, train_loader, val_loader, num_epochs=15, patience=4)

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Saved best model weights to {config.MODEL_SAVE_PATH}")

    plot_history(history)

    class_names = ["Cassava Bacterial Blight", "Cassava Brown Streak Disease",
                   "Cassava Green Mottle", "Cassava Mosaic Disease", "Healthy"]
    evaluate(model, val_loader, class_names)
