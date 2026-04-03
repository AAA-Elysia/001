import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据加载与预处理
digits = load_digits()
X = digits.data / 16.0  # 归一化到0-1
y = digits.target

# 划分训练集/验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# One-hot编码
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


y_train_onehot = one_hot(y_train, 10)


# Dropout层（Inverted Dropout）
class DropoutLayer:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            # 训练模式：生成掩码+缩放（核心）
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        else:
            # 测试模式：直接返回
            return x

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# 普通SGD优化器
class SGD:
    def __init__(self, lr=0.05):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


# Momentum SGD优化器
class MomentumSGD:
    def __init__(self, lr=0.05, gamma=0.9):
        self.lr = lr
        self.gamma = gamma
        self.velocities = {}  # 动量速度变量

    def update(self, params, grads):
        # 初始化速度变量
        if not self.velocities:
            for key in params.keys():
                self.velocities[key] = np.zeros_like(params[key])

        # 动量更新核心公式：v = γ*v - η*grad
        for key in params.keys():
            self.velocities[key] = self.gamma * self.velocities[key] - self.lr * grads[key]
            params[key] += self.velocities[key]


# MLP模型
class MLP:
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=10, dropout_p=0.2):
        # Xavier初始化参数
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(1 / input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * np.sqrt(1 / hidden_dim),
            'b2': np.zeros(output_dim)
        }
        self.dropout = DropoutLayer(dropout_p)

    def forward(self, x):
        # 前向传播：Linear→ReLU→Dropout→Linear→Softmax
        self.z1 = np.dot(x, self.params['W1']) + self.params['b1']
        self.a1 = np.maximum(0, self.z1)
        self.a1_drop = self.dropout.forward(self.a1)
        self.z2 = np.dot(self.a1_drop, self.params['W2']) + self.params['b2']

        # 数值稳定的Softmax
        exp_z2 = np.exp(self.z2 - np.max(self.z2, axis=1, keepdims=True))
        self.prob = exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)
        return self.prob

    def backward(self, x, y):
        # 反向传播计算梯度
        batch_size = x.shape[0]

        # 输出层梯度
        dz2 = self.prob - y
        dW2 = np.dot(self.a1_drop.T, dz2) / batch_size
        db2 = np.sum(dz2, axis=0) / batch_size

        # Dropout反向
        da1 = np.dot(dz2, self.params['W2'].T) * self.dropout.mask

        # 隐藏层梯度
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(x.T, dz1) / batch_size
        db1 = np.sum(dz1, axis=0) / batch_size

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def train(self):
        self.dropout.train()

    def eval(self):
        self.dropout.eval()


# 训练函数
def train_model(model, optimizer, X_train, y_train, X_val, y_val, epochs=100):
    train_loss_history = []
    val_acc_history = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        y_pred = model.forward(X_train)
        loss = -np.mean(np.sum(y_train * np.log(y_pred + 1e-8), axis=1))  # 交叉熵损失
        grads = model.backward(X_train, y_train)
        optimizer.update(model.params, grads)

        # 验证阶段
        model.eval()
        y_val_pred = model.forward(X_val)
        val_acc = accuracy_score(y_val, np.argmax(y_val_pred, axis=1))

        # 记录
        train_loss_history.append(loss)
        val_acc_history.append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d} | Train Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    return train_loss_history, val_acc_history


# 模型A：普通SGD + 无Dropout
model_A = MLP(dropout_p=0.0)
optimizer_A = SGD(lr=0.05)
print("=== 训练模型A（普通SGD + 无Dropout）===")
loss_A, acc_A = train_model(model_A, optimizer_A, X_train, y_train_onehot, X_val, y_val, epochs=100)

# 模型B：Momentum SGD + Dropout(p=0.5)
model_B = MLP(dropout_p=0.5)
optimizer_B = MomentumSGD(lr=0.05, gamma=0.9)
print("\n=== 训练模型B（Momentum SGD + Dropout p=0.5）===")
loss_B, acc_B = train_model(model_B, optimizer_B, X_train, y_train_onehot, X_val, y_val, epochs=100)

# 绘制并保存Loss曲线（核心对比图）
plt.figure(figsize=(10, 6))
plt.plot(loss_A, label='Model A (SGD + No Dropout)', linewidth=2)
plt.plot(loss_B, label='Model B (Momentum SGD + Dropout)', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Train Loss', fontsize=12)
plt.title('Train Loss vs Epoch', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('train_loss_comparison.png', dpi=300, bbox_inches='tight')
print("\nTrain Loss曲线已保存为: train_loss_comparison.png")

# 绘制并保存准确率曲线
plt.figure(figsize=(10, 6))
plt.plot(acc_A, label='Model A', linewidth=2)
plt.plot(acc_B, label='Model B', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('Validation Accuracy vs Epoch', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('val_acc_comparison.png', dpi=300, bbox_inches='tight')
print("Validation Accuracy曲线已保存为: val_acc_comparison.png")