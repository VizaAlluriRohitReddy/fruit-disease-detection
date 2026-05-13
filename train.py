import os
import torch
import torch.nn as nn
from tqdm import tqdm
import config
from dataset import get_dataloaders
from model import get_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(data_dir=None, save_path=None, num_classes=None):
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, class_names = \
        get_dataloaders(data_dir)

    # Set num classes
    if num_classes is None:
        num_classes = len(class_names)

    print(f"Training model for {num_classes} classes")

    # Load model
    model = get_model(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.NUM_EPOCHS
    )

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    # Save path
    if save_path is None:
        save_path = os.path.join(
            config.CHECKPOINT_DIR, "best_model.pth"
        )

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 40)

        # ── Training phase ──────────────────────────
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(train_loader,
                                   desc="Training"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total        += labels.size(0)
            correct      += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ── Validation phase ────────────────────────
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader,
                                       desc="Validating"):
                images  = images.to(device)
                labels  = labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted  = outputs.max(1)
                total        += labels.size(0)
                correct      += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc  = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | "
              f"Val   Acc: {val_acc:.2f}%")

        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
                'num_classes': num_classes
            }, save_path)
            print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%")

    print(f"\nTraining complete!")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")

    # Save history
    history = {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "train_accs":   train_accs,
        "val_accs":     val_accs
    }
    torch.save(
        history,
        os.path.join(config.RESULTS_DIR, "fruit_history.pth")
    )
    print("Training history saved!")

    return model, test_loader, class_names


if __name__ == "__main__":
    os.makedirs(
        os.path.join(
            os.path.dirname(__file__),
            "models", "fruit_model"
        ),
        exist_ok=True
    )
    train(
        data_dir=config.FRUIT_DATA_DIR,
        save_path=os.path.join(
            os.path.dirname(__file__),
            "models", "fruit_model", "best_model.pth"
        )
    )
