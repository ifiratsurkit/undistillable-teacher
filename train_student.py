import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.01
TEMPERATURE = 4.0
ALPHA = 0.5
MODEL_PATH = "checkpoints/teacher.pth"
STUDENT_SAVE_PATH = "checkpoints/student.pth"

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=2)

# Define models
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
def kd_loss(student_logits, teacher_logits, labels):
    T = TEMPERATURE
    alpha = ALPHA
    ce_loss = nn.CrossEntropyLoss()(student_logits, labels)
    kl_loss = nn.KLDivLoss(reduction="batchmean")(nn.functional.log_softmax(student_logits/T, dim=1),
                                                   nn.functional.softmax(teacher_logits/T, dim=1))
    return alpha * ce_loss + (1. - alpha) * kl_loss * T * T

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load(MODEL_PATH))
    teacher.eval()

    student = StudentNet().to(device)
    optimizer = optim.SGD(student.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(EPOCHS):
        student.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = kd_loss(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {running_loss/len(trainloader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), STUDENT_SAVE_PATH)
    print(f"Student model saved to {STUDENT_SAVE_PATH}")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = StudentNet().to(device)
    student.load_state_dict(torch.load(STUDENT_SAVE_PATH))
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

    print(f"Student Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    train()
    test()

