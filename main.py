import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time


# Define the Feedforward Neural Network
class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    model.to(device)
    train_loss_history, val_loss_history, val_accuracy_history = [], [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss_history.append(running_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss_history.append(val_loss / len(val_loader))
        val_accuracy_history.append(correct / total)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss_history[-1]:.4f}, "
              f"Val Loss: {val_loss_history[-1]:.4f}, "
              f"Val Accuracy: {val_accuracy_history[-1] * 100:.2f}%")

    return train_loss_history, val_loss_history, val_accuracy_history


# Plot training curves
def plot_training_curves(train_loss, val_loss, val_accuracy, title):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title(f"{title} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title(f"{title} - Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


# Plot confusion matrix
def plot_confusion_matrix(model, test_loader, device, labels):
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels_batch in test_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels_batch.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.show()


# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 64
    validation_split = 0.2

    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Split dataset into training and validation
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    cnn_model = SimpleCNN()
    fnn_model = FeedforwardNN()

    # Define loss and optimizers
    criterion = nn.CrossEntropyLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
    fnn_optimizer = optim.Adam(fnn_model.parameters(), lr=learning_rate)

    # Train and evaluate CNN
    print("Training CNN...")
    cnn_train_loss, cnn_val_loss, cnn_val_acc = train_model(
        cnn_model, train_loader, val_loader, criterion, cnn_optimizer, num_epochs, device
    )
    print("Evaluating CNN...")
    plot_confusion_matrix(cnn_model, test_loader, device, labels=list(range(10)))
    plot_training_curves(cnn_train_loss, cnn_val_loss, cnn_val_acc, "CNN")

    # Train and evaluate FNN
    print("Training FNN...")
    fnn_train_loss, fnn_val_loss, fnn_val_acc = train_model(
        fnn_model, train_loader, val_loader, criterion, fnn_optimizer, num_epochs, device
    )
    print("Evaluating FNN...")
    plot_confusion_matrix(fnn_model, test_loader, device, labels=list(range(10)))
    plot_training_curves(fnn_train_loss, fnn_val_loss, fnn_val_acc, "FNN")


if __name__ == "__main__":
    main()