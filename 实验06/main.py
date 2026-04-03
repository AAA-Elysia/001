import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#
# 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))  # (N, 6, 28, 28)
        x = self.pool(x)  # (N, 6, 14, 14)
        x = self.relu(self.conv2(x))  # (N, 16, 10, 10)
        x = self.pool(x)  # (N, 16, 5, 5)
        x = self.flatten(x)  # (N, 400)
        x = self.relu(self.fc1(x))  # (N, 120)
        x = self.relu(self.fc2(x))  # (N, 84)
        x = self.fc3(x)  # (N, 10)
        return x


# 维度自检函数
def check_shape():
    model = LeNet5()
    x = torch.randn(1, 1, 28, 28)
    print("Input:", x.shape)

    x = model.relu(model.conv1(x))
    print("After Conv1 + ReLU:", x.shape)
    assert x.shape == (1, 6, 28, 28)

    x = model.pool(x)
    print("After Pool1:", x.shape)
    assert x.shape == (1, 6, 14, 14)

    x = model.relu(model.conv2(x))
    print("After Conv2 + ReLU:", x.shape)
    assert x.shape == (1, 16, 10, 10)

    x = model.pool(x)
    print("After Pool2:", x.shape)
    assert x.shape == (1, 16, 5, 5)

    x = model.flatten(x)
    print("After Flatten:", x.shape)
    assert x.shape == (1, 400)

    x = model.relu(model.fc1(x))
    print("After FC1 + ReLU:", x.shape)
    assert x.shape == (1, 120)

    x = model.relu(model.fc2(x))
    print("After FC2 + ReLU:", x.shape)
    assert x.shape == (1, 84)

    x = model.fc3(x)
    print("After FC3:", x.shape)
    assert x.shape == (1, 10)

    print("All shapes are correct!")


# 运行维度检查
check_shape()

# 训练 Pipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

for epoch in range(num_epochs):
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

    train_loss = running_loss / len(train_loader.dataset)

    # 评估测试集准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = correct / total

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")