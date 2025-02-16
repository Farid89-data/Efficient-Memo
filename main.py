import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from efficientnet_pytorch import EfficientNet
import numpy as np
from sklearn.metrics import accuracy_score
import os
data_dir = "C:/Users/mehr110/PycharmProjects/Efficient-Memo/dataset/kaggle_data"
from torch.utils.data import DataLoader, random_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Dataset Handling
def load_kaggle_dataset(data_dir, img_size=224, batch_size=32):
    """
    Load and preprocess the Kaggle Agricultural Pest dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, len(dataset.classes)


# 2. Efficient Memo Model
class EfficientMemo(nn.Module):
    """
    Efficient Memo architecture using EfficientNet as backbone.
    """

    def __init__(self, num_classes, base_model="efficientnet-b0"):
        super(EfficientMemo, self).__init__()
        self.base_model = EfficientNet.from_pretrained(base_model)
        self.base_model._fc = nn.Identity()  # Remove final classification layer

        # Generalized layers
        self.generalized = nn.Sequential(*list(self.base_model.children())[:-1])

        # Specialized layers
        self.specialized = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),  # Example specialized layer
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.fc = nn.Linear(512, num_classes)  # Final classification layer

    def forward(self, x):
        x = self.generalized(x)  # Generalized features
        x = self.specialized(x)  # Specialized features
        x = torch.flatten(x, 1)  # Flatten for FC layer
        x = self.fc(x)  # Classification
        return x


# 3. Training Loop
def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """
    Train the model incrementally.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_efficient_memo.pth")

    print(f"Best Validation Accuracy: {best_acc:.4f}")
    return model


# 4. Incremental Learning
def incremental_learning(train_loaders, val_loaders, num_classes, base_model="efficientnet-b0"):
    """
    Train the model incrementally with new classes.
    """
    model = EfficientMemo(num_classes=num_classes, base_model=base_model)

    for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        print(f"\nTraining Session {i + 1}:")
        model = train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001)
        # Freeze specialized layers after each session
        for param in model.specialized.parameters():
            param.requires_grad = False

    return model


# 5. Main
if __name__ == "__main__":
    data_dir = "./kaggle_data"  # Path to Kaggle dataset
    train_loader, val_loader, num_classes = load_kaggle_dataset(data_dir)

    # Simulate incremental learning with 2 classes per session
    train_loaders, val_loaders = [train_loader], [val_loader]

    # Train the model incrementally
    final_model = incremental_learning(train_loaders, val_loaders, num_classes=num_classes)