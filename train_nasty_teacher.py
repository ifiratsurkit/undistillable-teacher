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
MODEL_PATH = "checkpoints/nasty_teacher.pth"

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

# Define nasty teacher model
class NastyTeacherNet(nn.Module):
    def __init__(self):
        super(NastyTeacherNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)

    def forward(self, x):
        return self.model(x)

# Nasty loss: cross-entropy + regularization to mislead distillation
class NastyLoss(nn.Module):
    def __init__(self):
        super(NastyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        ce_loss = self.ce(outputs, labels)
        softmax = nn.functional.softmax(outputs, dim=1)

        # Encourage high similarity with incorrect classes (misleading signals)
        batch_size = outputs.size(0)
        incorrect_mask = torch.ones_like(softmax)
        incorrect_mask[range(batch_size), labels] = 0
        incorrect_confidence = (softmax * incorrect_mask).sum(dim=1).mean()

        # Total loss: CE + lambda * misleading regularization
        lambda_reg = 0.5
        return ce_loss + lambda_reg * incorrect_confidence

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NastyTeacherNet().to(device)
    criterion = NastyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Nasty Teacher] Epoch {epoch+1} loss: {running_loss/len(trainloader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Nasty teacher model saved to {MODEL_PATH}")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NastyTeacherNet().to(device)
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

    print(f"Nasty Teacher Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    train()
    test()

