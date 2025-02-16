import torch
from torchvision import transforms
from model import MemoBase
from dataset import IncrementalDataset
from trainer import MemoTrainer


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = MemoBase(backbone_type='efficientnet', num_classes=2).to(device)

    # Setup trainer
    trainer = MemoTrainer(model, device)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Training sessions
    num_sessions = 5
    classes_per_session = 2

    for session in range(num_sessions):
        print(f"\nTraining session {session + 1}")

        # Create datasets for current session
        train_dataset = IncrementalDataset(
            root_dir='C:/Users/mehr110/PycharmProjects/Efficient-Memo/dataset/kaggle_data',
            class_range=range(session * classes_per_session, (session + 1) * classes_per_session),
            transform=transform
        )

        val_dataset = IncrementalDataset(
            root_dir='C:/Users/mehr110/PycharmProjects/Efficient-Memo/dataset/kaggle_data',
            class_range=range(session * classes_per_session, (session + 1) * classes_per_session),
            transform=transform
        )

        # Expand model for new classes (except first session)
        if session > 0:
            model.expand_model(classes_per_session)

        # Train for current session
        acc = trainer.train_session(train_dataset, val_dataset)
        print(f"Session {session + 1} finished with best accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
