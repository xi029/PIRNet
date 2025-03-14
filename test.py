from ultralytics import YOLO
import torch
from torchvision import datasets, transforms
from collections import defaultdict

def evaluate_model(model, data_loader, device):
    model.to(device)
    model.eval()

    # Initialize counters for each class
    correct_by_class = defaultdict(int)
    total_by_class = defaultdict(int)
    total_correct = 0
    total_samples = 0

    # Loop through the test dataset
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions
            predictions = model(images).argmax(dim=1)

            # Update overall correct predictions
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            # Update counters per class
            for label, prediction in zip(labels, predictions):
                total_by_class[label.item()] += 1
                if prediction == label:
                    correct_by_class[label.item()] += 1

    # Calculate total accuracy
    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    # Calculate per-class accuracy
    class_accuracy = {
        cls: (correct_by_class[cls] / total_by_class[cls]) if total_by_class[cls] > 0 else 0
        for cls in total_by_class.keys()
    }

    return total_accuracy, class_accuracy

if __name__ == "__main__":
    # Load the trained model
    model = YOLO("runs/classify/train5/weights/best.pt")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_data_dir = r"D:\yolo_paper\ultralytics-main\datasets\test"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ensure image size matches model input
        transforms.ToTensor()
    ])
    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    total_accuracy, class_accuracy = evaluate_model(model, test_loader, device)

    # Output results
    print(f"Total Accuracy: {total_accuracy:.2f}")
    for cls, acc in class_accuracy.items():
        print(f"Class {cls} Accuracy: {acc:.2f}")
