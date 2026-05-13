import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import urllib.request
import os

FRUIT_CLASSES = [
    'apple', 'banana', 'orange', 'mango', 'grape',
    'strawberry', 'tomato', 'potato', 'pepper', 'guava',
    'pomegranate', 'cucumber', 'carrot', 'lemon', 'lime',
    'watermelon', 'pineapple', 'pear', 'peach', 'plum',
    'cherry', 'blueberry', 'raspberry', 'fig', 'coconut',
    'fruit', 'vegetable', 'plant', 'leaf', 'flower',
    'herb', 'tree', 'bush', 'vine', 'crop',
    'hop', 'berry', 'citrus', 'tropical', 'root',
    'gourd', 'melon', 'seed', 'ripe', 'fresh',
    'produce', 'fungus', 'mold', 'rot', 'blight',
    'spot', 'rust', 'wilt', 'scab', 'lesion'
]

NON_FRUIT_CLASSES = [
    'logo', 'icon', 'sign', 'symbol', 'badge',
    'ball', 'car', 'dog', 'cat', 'person',
    'building', 'computer', 'phone', 'book',
    'chair', 'table', 'keyboard', 'monitor',
    'jersey', 'uniform', 'flag', 'toy'
]


def get_imagenet_classes():
    classes_path = os.path.join(
        os.path.dirname(__file__), 'imagenet_classes.txt'
    )
    if not os.path.exists(classes_path):
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        urllib.request.urlretrieve(url, classes_path)
    with open(classes_path) as f:
        return [line.strip() for line in f.readlines()]


class FruitValidator:
    def __init__(self):
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
        self.model.eval()
        self.classes = get_imagenet_classes()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def is_fruit_or_leaf(self, image):
        tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            top5_probs, top5_indices = probs.topk(5)

        top5_classes = [
            self.classes[idx].lower()
            for idx in top5_indices[0]
        ]
        top5_prob_values = [
            probs[0][idx].item()
            for idx in top5_indices[0]
        ]

        print(f"Top 5 classes: {top5_classes}")
        print(f"Top 5 probs: {top5_prob_values}")

        # Check if any non-fruit class detected strongly
        for i, cls in enumerate(top5_classes):
            for non_fruit in NON_FRUIT_CLASSES:
                if non_fruit in cls and top5_prob_values[i] > 0.3:
                    return False, cls

        # Check if any fruit class detected
        for cls in top5_classes:
            for fruit in FRUIT_CLASSES:
                if fruit in cls:
                    return True, cls

        return False, top5_classes[0]


validator = None


def get_validator():
    global validator
    if validator is None:
        validator = FruitValidator()
    return validator


def validate_image(image):
    v = get_validator()
    is_fruit, detected_class = v.is_fruit_or_leaf(image)
    return is_fruit, detected_class
