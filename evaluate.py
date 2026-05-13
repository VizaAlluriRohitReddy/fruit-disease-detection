import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import config
from dataset import get_dataloaders
from model import get_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate():
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    _, _, test_loader, class_names = get_dataloaders()

    # Load best model
    model = get_model().to(device)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print("Best model loaded!")

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images  = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report
    print("\n── Classification Report ──────────────────")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names
    ))

    # Overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = 100.0 * correct / len(all_labels)
    print(f"Test Accuracy: {acc:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 16))
    sns.heatmap(
        cm,
        annot=False,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")

    # Plot training history
    history_path = os.path.join(config.RESULTS_DIR, "history.pth")
    if os.path.exists(history_path):
        history = torch.load(history_path)
        plot_history(history)


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history["train_losses"], label="Train Loss")
    ax1.plot(history["val_losses"],   label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(history["train_accs"], label="Train Accuracy")
    ax2.plot(history["val_accs"],   label="Val Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")


if __name__ == "__main__":
    evaluate()
