import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X, y = data.data, data.target

# 划分训练集/测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 输出数据维度
print("数据维度信息：")
print(f"训练集 X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"测试集 X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

def sigmoid(z):
    # Sigmoid函数，处理数值稳定性避免溢出
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

# 测试sigmoid函数
print("\n=== 测试sigmoid函数 ===")
print(f"sigmoid(0) = {sigmoid(0)}")

def predict_proba(X, w, b):
    # 预测正类概率：p = σ(X·w + b)
    z = np.dot(X, w) + b
    p = sigmoid(z)
    return p

def cross_entropy_loss(y_true, y_pred):
    # 二元交叉熵损失，添加ε防止log(0)
    epsilon = 1e-10
    loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return loss

def compute_gradient(X, y_true, y_pred):
    # 计算权重和偏置的梯度
    n_samples = X.shape[0]
    dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_true))
    db = (1 / n_samples) * np.sum(y_pred - y_true)
    return dw, db

# 梯度下降训练逻辑回归
def train_logistic_regression(X_train, y_train, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0.0
    loss_history = []

    # 训练循环
    for epoch in range(epochs):
        # 前向传播
        y_pred = predict_proba(X_train, w, b)
        # 计算损失
        loss = cross_entropy_loss(y_train, y_pred)
        loss_history.append(loss)
        # 计算梯度
        dw, db = compute_gradient(X_train, y_train, y_pred)
        w -= learning_rate * dw
        b -= learning_rate * db

        # 每100轮打印损失
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")

    return w, b, loss_history

# 设置超参数
LEARNING_RATE = 0.01
EPOCHS = 1000
print("\n=== 开始训练逻辑回归 ===")
w_trained, b_trained, loss_history = train_logistic_regression(
    X_train, y_train, learning_rate=LEARNING_RATE, epochs=EPOCHS
)

# 模型评估
def predict(X, w, b, threshold=0.5):
    y_pred_proba = predict_proba(X, w, b)
    y_pred = (y_pred_proba >= threshold).astype(int)
    return y_pred

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# 预测与评估
y_test_pred = predict(X_test, w_trained, b_trained)
test_accuracy = accuracy(y_test, y_test_pred)
print(f"测试集准确率: {test_accuracy:.4f}")

# 画图
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, color='blue', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.title('Logistic Regression Training Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig("logistic_regression_loss_curve.png")
plt.show()