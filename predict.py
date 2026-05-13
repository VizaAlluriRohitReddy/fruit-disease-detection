import os
import sys
import torch
from PIL import Image
from torchvision import transforms
import config
from model import get_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict(image_path):
    device = get_device()

    # Load class names
    from torchvision.datasets import ImageFolder
    dataset    = ImageFolder(root=config.DATA_DIR)
    class_names = dataset.classes

    # Load model
    model = get_model().to(device)
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print("No trained model found!")
        print("Please run train.py first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs     = model(image)
        probs       = torch.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)

    predicted_class = class_names[predicted.item()]
    confidence_pct  = confidence.item() * 100

    print(f"\n── Prediction Result ──────────────────────")
    print(f"Image      : {image_path}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence_pct:.2f}%")

    # Top 3 predictions
    top3_probs, top3_indices = probs.topk(3, dim=1)
    print(f"\nTop 3 Predictions:")
    for i in range(3):
        cls  = class_names[top3_indices[0][i].item()]
        prob = top3_probs[0][i].item() * 100
        print(f"  {i+1}. {cls} ({prob:.2f}%)")

    return predicted_class, confidence_pct


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image>")
        print("Example: python3 predict.py data/test_apple.jpg")
    else:
        predict(sys.argv[1])
