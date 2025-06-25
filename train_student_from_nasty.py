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
NASTY_TEACHER_PATH = "checkpoints/nasty_teacher.pth"
STUDENT_PATH = "checkpoints/student_from_nasty.pth"

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model definitions
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

class NastyTeacherNet(nn.Module):
    def __init__(self):
        super(NastyTeacherNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# KD loss function
def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha):
    kd_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_outputs / T, dim=1),
        nn.functional.softmax(teacher_outputs / T, dim=1)
    ) * (T * T)
    ce_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss

def train_student_from_nasty_teacher():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher = NastyTeacherNet().to(device)
    teacher.load_state_dict(torch.load(NASTY_TEACHER_PATH))
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

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), STUDENT_PATH)
    print(f"Student from nasty teacher saved to {STUDENT_PATH}")

    # Plotting results
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o')
    plt.title('Nasty Teacher → Student Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('nasty_loss.png')

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), accuracy_history, marker='o')
    plt.title('Nasty Teacher → Student Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig('nasty_accuracy.png')

if __name__ == "__main__":
    train_student_from_nasty_teacher()
