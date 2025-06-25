import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import os

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MODEL_PATH = "checkpoints/nasty_teacher.pth"

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Define nasty teacher model
class NastyTeacher(nn.Module):
    def __init__(self):
        super(NastyTeacher, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

def adversarial_loss(outputs, labels):
    # Encourage high confidence for incorrect classes
    softmax = torch.softmax(outputs, dim=1)
    wrong_class_probs = softmax.clone()
    wrong_class_probs[range(len(labels)), labels] = 0  # Zero out correct class
    max_wrong = wrong_class_probs.max(dim=1)[0]  # Max probability among wrong classes
    return max_wrong.mean()  # Encourage these to be high

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NastyTeacher().to(device)
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss1 = ce_loss(outputs, labels)  # Main loss
            loss2 = adversarial_loss(outputs, labels)  # Sabotage loss
            loss = loss1 + 0.5 * loss2  # Weighted sum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Nasty Teacher model saved to {MODEL_PATH}")

    # Save plots
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.plot(range(1, EPOCHS+1), epoch_losses, label='Loss', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Nasty Teacher)")
    plt.grid(True)
    plt.savefig("outputs/nasty_teacher_loss.png")

    plt.figure()
    plt.plot(range(1, EPOCHS+1), epoch_accuracies, label='Accuracy', color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy (Nasty Teacher)")
    plt.grid(True)
    plt.savefig("outputs/nasty_teacher_accuracy.png")

    print("Training plots saved to outputs/")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NastyTeacher().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy (Nasty Teacher): {100 * correct / total:.2f}%")

if __name__ == '__main__':
    train()
    test()
