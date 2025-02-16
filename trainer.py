import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class MemoTrainer:
    def __init__(self, model, device, num_epochs=100, batch_size=32):
        self.model = model
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def _setup_optimizer(self):
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4
        )

    def train_session(self, train_dataset, val_dataset):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        self._setup_optimizer()
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            # Validation phase
            self.model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            print(
                f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {running_loss / len(train_loader):.3f}, Accuracy: {accuracy:.2f}%')

            if accuracy > best_acc:
                best_acc = accuracy
                # Save best model

        return best_acc