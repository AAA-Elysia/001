import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======================================
# 1. 数据加载与预处理
# ======================================
# 加载手写数字数据集
digits = load_digits()
X, y = digits.data, digits.target

# 划分训练集/验证集/测试集（60/20/20，分层抽样保证类别分布）
# 第一步：划分训练集(60%)和临时集(40%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
# 第二步：划分验证集(20%)和测试集(20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# 标准化（逻辑回归/MLP对特征尺度敏感，必须标准化）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 仅用训练集拟合均值/方差
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 输出数据维度
print("=== 数据维度信息 ===")
print(f"训练集 X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"验证集 X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
print(f"测试集 X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")


# ======================================
# 2. 核心函数实现（手动前向/反向传播）
# ======================================
def relu(x):
    """实现ReLU激活函数：relu(x) = max(0, x)"""
    return np.maximum(0, x)


def relu_derivative(x):
    """ReLU的导数：x>0时为1，x≤0时为0"""
    return np.where(x > 0, 1, 0)


def softmax(z):
    """
    实现Softmax函数（处理数值稳定性：减去每行最大值）
    z: (B, C) 原始得分
    返回：(B, C) 概率分布
    """
    z_max = np.max(z, axis=1, keepdims=True)  # (B, 1)
    exp_z = np.exp(z - z_max)  # 防止指数溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(p, y_true):
    """
    多分类交叉熵损失
    p: (B, C) 预测概率
    y_true: (B,) 真实标签（整数）
    返回：标量 平均损失
    """
    epsilon = 1e-10  # 防止log(0)
    # 取出每个样本对应真实类别的概率
    p_true = p[np.arange(len(y_true)), y_true]
    loss = -np.mean(np.log(p_true + epsilon))
    return loss


def init_params(D=64, H=128, C=10):
    """
    初始化参数
    W1: (D, H) 第一层权重，正态分布N(0, 0.01)
    b1: (H,) 第一层偏置，全0
    W2: (H, C) 第二层权重，正态分布N(0, 0.01)
    b2: (C,) 第二层偏置，全0
    """
    np.random.seed(42)  # 固定随机种子，结果可复现
    W1 = np.random.normal(0, 0.01, (D, H))
    b1 = np.zeros(H)
    W2 = np.random.normal(0, 0.01, (H, C))
    b2 = np.zeros(C)

    # 打印参数形状
    print("\n=== 参数初始化形状 ===")
    print(f"W1.shape: {W1.shape}, b1.shape: {b1.shape}")
    print(f"W2.shape: {W2.shape}, b2.shape: {b2.shape}")
    return W1, b1, W2, b2


def forward(Xb, W1, b1, W2, b2):
    """
    前向传播
    Xb: (B, D) batch输入
    返回：Z1, A1, Z2, P（中间变量，用于反向传播）
    """
    # 第一层：Linear -> ReLU
    Z1 = np.dot(Xb, W1) + b1  # (B, H)
    A1 = relu(Z1)  # (B, H)
    # 第二层：Linear -> Softmax
    Z2 = np.dot(A1, W2) + b2  # (B, C)
    P = softmax(Z2)  # (B, C)
    return Z1, A1, Z2, P


def backward(Xb, y_true, Z1, A1, Z2, P, W2):
    """
    反向传播（手动推导梯度）
    返回：dW1, db1, dW2, db2
    """
    B = Xb.shape[0]  # batch size

    # 步骤1：计算dZ2（Z2的梯度）
    # dZ2 = P - y_onehot (B, C)
    y_onehot = np.zeros_like(P)
    y_onehot[np.arange(B), y_true] = 1
    dZ2 = (P - y_onehot) / B  # 除以B，后续梯度不用再除

    # 步骤2：计算dW2, db2
    dW2 = np.dot(A1.T, dZ2)  # (H, C)
    db2 = np.sum(dZ2, axis=0)  # (C,)

    # 步骤3：计算dA1, dZ1
    dA1 = np.dot(dZ2, W2.T)  # (B, H)
    dZ1 = dA1 * relu_derivative(Z1)  # (B, H)

    # 步骤4：计算dW1, db1
    dW1 = np.dot(Xb.T, dZ1)  # (D, H)
    db1 = np.sum(dZ1, axis=0)  # (H,)

    # 打印梯度形状
    print("\n=== 梯度形状（首次反向传播） ===")
    print(f"dW1.shape: {dW1.shape}, db1.shape: {db1.shape}")
    print(f"dW2.shape: {dW2.shape}, db2.shape: {db2.shape}")
    return dW1, db1, dW2, db2


# ======================================
# 3. Mini-batch 梯度下降训练
# ======================================
def create_mini_batches(X, y, batch_size=64, shuffle=True):
    """创建mini-batch生成器"""
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(X), batch_size):
        end_idx = min(start_idx + batch_size, len(X))
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def train_mlp(X_train, y_train, X_val, y_val,
              batch_size=64, lr=0.01, epochs=50):
    """
    训练两层MLP
    返回：训练好的参数 + 损失/准确率历史
    """
    # 初始化参数
    D, H, C = 64, 128, 10
    W1, b1, W2, b2 = init_params(D, H, C)

    # 记录训练过程
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    # 训练循环
    for epoch in range(epochs):
        # ----------------------
        # 训练阶段（mini-batch梯度下降）
        # ----------------------
        train_loss = 0.0
        batch_count = 0
        # 遍历所有mini-batch
        for Xb, yb in create_mini_batches(X_train, y_train, batch_size):
            # 前向传播
            Z1, A1, Z2, P = forward(Xb, W1, b1, W2, b2)
            # 计算batch损失
            loss = cross_entropy_loss(P, yb)
            train_loss += loss
            batch_count += 1
            # 反向传播
            dW1, db1, dW2, db2 = backward(Xb, yb, Z1, A1, Z2, P, W2)
            # 更新参数（仅首次反向传播打印梯度形状，后续省略）
            if epoch == 0 and batch_count == 1:
                pass  # 已打印过梯度形状
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # 计算本轮平均训练损失
        avg_train_loss = train_loss / batch_count
        train_loss_history.append(avg_train_loss)

        # ----------------------
        # 验证阶段
        # ----------------------
        # 验证集前向传播
        _, _, _, P_val = forward(X_val, W1, b1, W2, b2)
        val_loss = cross_entropy_loss(P_val, y_val)
        val_loss_history.append(val_loss)
        # 计算验证集准确率
        val_pred = np.argmax(P_val, axis=1)
        val_acc = np.mean(val_pred == y_val)
        val_acc_history.append(val_acc)

        # 打印本轮结果
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch [{epoch + 1}/{epochs}]")
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return W1, b1, W2, b2, train_loss_history, val_loss_history, val_acc_history


# ======================================
# 4. 模型评估（测试集）
# ======================================
def evaluate(X_test, y_test, W1, b1, W2, b2):
    """评估测试集准确率"""
    _, _, _, P_test = forward(X_test, W1, b1, W2, b2)
    test_pred = np.argmax(P_test, axis=1)
    test_acc = np.mean(test_pred == y_test)
    print("\n=== 测试集评估结果 ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc


# ======================================
# 5. 绘制训练曲线
# ======================================
def plot_curves(train_loss, val_loss, val_acc, epochs):
    """绘制损失曲线和准确率曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    ax1.plot(range(1, epochs + 1), train_loss, label='Train Loss', color='blue')
    ax1.plot(range(1, epochs + 1), val_loss, label='Val Loss', color='red')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cross Entropy Loss')
    ax1.set_title('Training & Validation Loss Curve')
    ax1.legend()
    ax1.grid(True)

    # 验证集准确率曲线
    ax2.plot(range(1, epochs + 1), val_acc, label='Val Accuracy', color='green')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy Curve')
    ax2.legend()
    ax2.grid(True)

    # 保存图片
    plt.tight_layout()
    plt.savefig("mlp_training_curves.png")
    plt.show()


# ======================================
# 主程序：运行训练和评估
# ======================================
if __name__ == "__main__":
    # 超参数设置
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    EPOCHS = 50

    # 训练模型
    print("\n=== 开始训练两层MLP ===")
    W1_trained, b1_trained, W2_trained, b2_trained, train_loss, val_loss, val_acc = train_mlp(
        X_train, y_train, X_val, y_val,
        batch_size=BATCH_SIZE, lr=LEARNING_RATE, epochs=EPOCHS
    )

    # 评估测试集
    test_acc = evaluate(X_test, y_test, W1_trained, b1_trained, W2_trained, b2_trained)

    # 绘制曲线
    plot_curves(train_loss, val_loss, val_acc, EPOCHS)