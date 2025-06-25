import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
TEMPERATURE = 5.0
ALPHA = 0.7
MODEL_PATH = "checkpoints/student.pth"
TEACHER_PATH = "checkpoints/teacher.pth"

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Define student model
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Load pre-trained teacher model
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Soft target loss (KL Divergence) + Hard label loss (CrossEntropy)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentNet().to(device)
    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH))
    teacher.eval()

    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0.9)

    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels, TEMPERATURE, ALPHA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Save student model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), MODEL_PATH)
    print(f"Student model saved to {MODEL_PATH}")

    # Save plots
    os.makedirs("outputs", exist_ok=True)

    plt.figure()
    plt.plot(range(1, EPOCHS+1), epoch_losses, label='Loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss (Student)")
    plt.grid(True)
    plt.savefig("outputs/student_loss.png")

    plt.figure()
    plt.plot(range(1, EPOCHS+1), epoch_accuracies, label='Accuracy', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Accuracy (Student)")
    plt.grid(True)
    plt.savefig("outputs/student_accuracy.png")

    print("Training plots saved to outputs/")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudentNet().to(device)
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

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    train()
    test()
