import argparse
import csv
import random
import sys
from pathlib import Path


LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_stripe_image(label, image_size, rng):
    image = Image.new("RGB", (image_size, image_size), (20, 20, 20))
    draw = ImageDraw.Draw(image)

    stripe_width = int(rng.integers(image_size // 8, image_size // 5))
    offset = int(rng.integers(-image_size // 8, image_size // 8))
    color = (
        int(rng.integers(190, 255)),
        int(rng.integers(190, 255)),
        int(rng.integers(190, 255)),
    )

    if label == "vertical":
        center = image_size // 2 + offset
        draw.rectangle(
            [center - stripe_width // 2, 0, center + stripe_width // 2, image_size],
            fill=color,
        )
    else:
        center = image_size // 2 + offset
        draw.rectangle(
            [0, center - stripe_width // 2, image_size, center + stripe_width // 2],
            fill=color,
        )

    noise = rng.integers(0, 15, size=(image_size, image_size, 3), dtype=np.int16)
    arr = np.asarray(image, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_dataset(root, image_size=64, train_per_class=40, val_per_class=10, test_per_class=10, seed=7):
    rng = np.random.default_rng(seed)
    labels = ["horizontal", "vertical"]
    splits = {
        "train": train_per_class,
        "val": val_per_class,
        "test": test_per_class,
    }

    root = Path(root)
    for split, count in splits.items():
        for label in labels:
            label_dir = root / split / label
            label_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                image = make_stripe_image(label, image_size, rng)
                image.save(label_dir / f"{label}_{i:03d}.png")


class ImageFolderDataset(Dataset):
    def __init__(self, root, image_size=64, class_to_idx=None):
        self.root = Path(root)
        self.image_size = image_size

        if class_to_idx is None:
            classes = sorted([path.name for path in self.root.iterdir() if path.is_dir()])
            self.class_to_idx = {name: i for i, name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root / class_name
            for image_path in sorted(class_dir.glob("*.png")):
                self.samples.append((image_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, label


def scaled_channels(scale):
    base_channels = [64, 128, 256, 512, 512]
    return [max(8, int(value * scale)) for value in base_channels]


class VGG19(nn.Module):
    def __init__(self, num_classes=2, channel_scale=0.125, classifier_width=128, use_batch_norm=True):
        super().__init__()
        self.use_batch_norm = use_batch_norm
        c1, c2, c3, c4, c5 = scaled_channels(channel_scale)
        config = [
            c1, c1, "M",
            c2, c2, "M",
            c3, c3, c3, c3, "M",
            c4, c4, c4, c4, "M",
            c5, c5, c5, c5, "M",
        ]

        self.features = self._make_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(c5 * 2 * 2, classifier_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_width, classifier_width),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(classifier_width, num_classes),
        )

    def _make_features(self, config):
        layers = []
        in_channels = 3
        for item in config:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, item, kernel_size=3, padding=1))
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm2d(item))
                layers.append(nn.ReLU(inplace=True))
                in_channels = item
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_predictions, all_labels


def classification_metrics(predictions, labels, num_classes):
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, predicted_label in zip(labels, predictions):
        confusion[true_label][predicted_label] += 1

    per_class = []
    for class_index in range(num_classes):
        tp = confusion[class_index][class_index]
        fp = sum(confusion[row][class_index] for row in range(num_classes) if row != class_index)
        fn = sum(confusion[class_index][col] for col in range(num_classes) if col != class_index)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class.append((precision, recall, f1))

    accuracy = sum(confusion[i][i] for i in range(num_classes)) / len(labels)
    macro_precision = sum(item[0] for item in per_class) / num_classes
    macro_recall = sum(item[1] for item in per_class) / num_classes
    macro_f1 = sum(item[2] for item in per_class) / num_classes

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": confusion,
    }


def save_history(path, history):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for row in history:
            writer.writerow(row)


def save_report(path, metrics, class_names, parameter_count, args):
    with open(path, "w", encoding="utf-8") as f:
        f.write("Labwork 6 - CNN Evaluation Report\n")
        f.write("=================================\n\n")
        f.write(f"Model: VGG19 block structure, channel_scale={args.channel_scale}\n")
        f.write(f"Trainable parameters: {parameter_count}\n")
        f.write(f"Image size: {args.image_size}x{args.image_size}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n\n")
        f.write("Overall metrics\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"F1-score: {metrics['macro_f1']:.4f}\n\n")
        f.write("Per-class metrics\n")
        for class_name, values in zip(class_names, metrics["per_class"]):
            precision, recall, f1 = values
            f.write(f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}\n")
        f.write("\nConfusion matrix rows=true labels, columns=predicted labels\n")
        for row in metrics["confusion"]:
            f.write(" ".join(str(value) for value in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Labwork 6: VGG19 CNN image classification")
    parser.add_argument("--data-dir", default="Lab6/data/stripe_dataset")
    parser.add_argument("--output-dir", default="Lab6/output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channel-scale", type=float, default=0.125)
    parser.add_argument("--classifier-width", type=int, default=128)
    parser.add_argument("--no-batch-norm", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-generate-data", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_generate_data:
        generate_dataset(data_dir, image_size=args.image_size, seed=args.seed)

    train_dataset = ImageFolderDataset(data_dir / "train", image_size=args.image_size)
    val_dataset = ImageFolderDataset(
        data_dir / "val",
        image_size=args.image_size,
        class_to_idx=train_dataset.class_to_idx,
    )
    test_dataset = ImageFolderDataset(
        data_dir / "test",
        image_size=args.image_size,
        class_to_idx=train_dataset.class_to_idx,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = VGG19(
        num_classes=len(train_dataset.class_to_idx),
        channel_scale=args.channel_scale,
        classifier_width=args.classifier_width,
        use_batch_norm=not args.no_batch_norm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = count_parameters(model)

    print("VGG19 CNN training")
    print(f"Classes: {train_dataset.class_to_idx}")
    print(f"Trainable parameters: {parameter_count}")
    print(f"Device: {device}")

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        history.append([epoch, train_loss, train_accuracy, val_loss, val_accuracy])
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    test_loss, test_accuracy, predictions, labels = evaluate(model, test_loader, criterion, device)
    metrics = classification_metrics(predictions, labels, len(train_dataset.class_to_idx))
    class_names = [name for name, _ in sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1])]

    print("\nTest result")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall: {metrics['macro_recall']:.4f}")
    print(f"F1-score: {metrics['macro_f1']:.4f}")
    print("Confusion matrix:")
    for row in metrics["confusion"]:
        print(row)

    save_history(output_dir / "history.csv", history)
    save_report(output_dir / "classification_report.txt", metrics, class_names, parameter_count, args)
    torch.save(model.state_dict(), output_dir / "vgg19_cnn_state.pt")


if __name__ == "__main__":
    main()
