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
TEMPERATURE = 4.0
ALPHA = 0.7
TEACHER_PATH = "checkpoints/teacher.pth"
STUDENT_PATH = "checkpoints/student.pth"

# CIFAR-10 Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model Definitions
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Knowledge Distillation Loss
def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha):
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_outputs / T, dim=1),
        nn.functional.softmax(teacher_outputs / T, dim=1)
    ) * (T * T)
    ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

def train_student():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH))
    teacher.eval()

    student = StudentNet().to(device)
    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0.9)

    loss_history = []
    accuracy_history = []

    for epoch in range(EPOCHS):
        student.train()
        total_loss = 0.0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels, T=TEMPERATURE, alpha=ALPHA)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        loss_history.append(avg_loss)

        # Evaluation
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = student(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_history.append(accuracy)

        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Test Accuracy = {accuracy:.2f}%")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), STUDENT_PATH)
    print(f"Student model saved to {STUDENT_PATH}")

    # Plot results
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o')
    plt.title('Student Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('student_loss.png')

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), accuracy_history, marker='o')
    plt.title('Student Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig('student_accuracy.png')

if __name__ == "__main__":
    train_student()
