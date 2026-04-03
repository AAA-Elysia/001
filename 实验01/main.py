import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("数据维度信息：")
print(f"训练集 X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"验证集 X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
print(f"测试集 X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 定义两层MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # CrossEntropyLoss自动处理softmax

# 初始化模型、损失函数、优化器
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 20
train_loss_history = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    train_steps = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_steps += 1

    # 计算平均训练损失
    avg_train_loss = total_train_loss / train_steps
    train_loss_history.append(avg_train_loss)
    print(f"\nEpoch [{epoch+1}/{epochs}]")
    print(f"平均训练损失: {avg_train_loss:.4f}")

    # 验证
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    val_accuracy = correct / total
    print(f"验证集准确率: {val_accuracy:.4f}")

# 评估
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

test_accuracy = correct / total
print(f"测试集准确率: {test_accuracy:.4f}")

# 保存/加载模型
torch.save(model.state_dict(), "mlp_digits_model.pth")
print("\n模型已保存为 mlp_digits_model.pth")

# 加载模型并验证
new_model = MLP().to(device)
new_model.load_state_dict(torch.load("mlp_digits_model.pth"))

new_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = new_model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

new_test_accuracy = correct / total
print(f"加载模型后测试集准确率: {new_test_accuracy:.4f}")

# 画图
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_loss_history, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Average Train Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("train_loss_curve.png")
plt.show()