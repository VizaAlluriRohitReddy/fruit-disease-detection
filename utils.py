import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import config


def verify_dataset(train_loader, class_names):
    """Show a sample batch of images to verify dataset loads correctly."""
    images, labels = next(iter(train_loader))

    # Unnormalize images for display
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    plt.figure(figsize=(16, 4))
    for i in range(min(8, len(images))):
        img = images[i].numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.subplot(1, 8, i + 1)
        plt.imshow(img)
        plt.title(class_names[labels[i]], fontsize=6)
        plt.axis("off")

    plt.suptitle("Sample Training Images", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "sample_images.png")
    plt.savefig(save_path, dpi=150)
    print(f"Sample images saved to {save_path}")


def print_model_summary(model):
    """Print total number of parameters."""
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"\n── Model Summary ───────────────────────────")
    print(f"Total parameters     : {total/1e6:.2f} Million")
    print(f"Trainable parameters : {trainable/1e6:.2f} Million")
