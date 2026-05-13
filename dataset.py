import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import config


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def get_dataloaders(data_dir=None):
    if data_dir is None:
        data_dir = config.DATA_DIR

    train_transform, val_test_transform = get_transforms()

    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=train_transform
    )

    total = len(full_dataset)
    train_size = int(config.TRAIN_SPLIT * total)
    val_size   = int(config.VAL_SPLIT   * total)
    test_size  = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_set.dataset.transform  = val_test_transform
    test_set.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    class_names = full_dataset.classes
    print(f"Dataset      : {data_dir}")
    print(f"Total images : {total}")
    print(f"Training     : {train_size}")
    print(f"Validation   : {val_size}")
    print(f"Testing      : {test_size}")
    print(f"Classes      : {len(class_names)}")
    print(f"Class names  : {class_names}")

    return train_loader, val_loader, test_loader, class_names
