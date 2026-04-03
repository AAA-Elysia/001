import numpy as np
import matplotlib.pyplot as plt

# ===================== 手写 2D 卷积 =====================
def my_conv2d(img, kernel, stride=1, padding=0):
    H, W = img.shape
    K, _ = kernel.shape
    padded_img = np.pad(img, ((padding, padding), (padding, padding)), mode='constant')

    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1
    assert H_out > 0 and W_out > 0, "输出维度不能为负数，请检查参数！"

    output = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            start_h = i * stride
            start_w = j * stride
            region = padded_img[start_h:start_h+K, start_w:start_w+K]
            output[i, j] = np.sum(region * kernel)
    return output

# ===================== 手写最大池化 =====================
def my_maxpool2d(img, kernel_size=2, stride=2):
    H, W = img.shape
    K = kernel_size
    H_out = (H - K) // stride + 1
    W_out = (W - K) // stride + 1
    assert H_out > 0 and W_out > 0, "输出维度不能为负数！"

    output = np.zeros((H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            start_h = i * stride
            start_w = j * stride
            region = img[start_h:start_h+K, start_w:start_w+K]
            output[i, j] = np.max(region)
    return output

# ===================== 主程序：零依赖生成测试图 =====================
if __name__ == "__main__":
    # 直接生成 128x128 灰度图，不依赖任何数据集！
    np.random.seed(123)
    img = np.linspace(0, 1, 128*128).reshape(128, 128)
    img = img + np.random.randn(128, 128) * 0.1
    img = np.clip(img, 0, 1)

    # Sobel 核
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    # 卷积
    edge_x = my_conv2d(img, sobel_x, stride=1, padding=1)
    edge_y = my_conv2d(img, sobel_y, stride=1, padding=1)

    # 池化
    pool_x = my_maxpool2d(edge_x, kernel_size=2, stride=2)
    pool_y = my_maxpool2d(edge_y, kernel_size=2, stride=2)

    # 画图
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image (128x128)')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(edge_x, cmap='gray')
    plt.title('Sobel X (Vertical Edge)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(edge_y, cmap='gray')
    plt.title('Sobel Y (Horizontal Edge)')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(pool_x, cmap='gray')
    plt.title(f'MaxPool X ({pool_x.shape[0]}x{pool_x.shape[1]})')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(pool_y, cmap='gray')
    plt.title(f'MaxPool Y ({pool_y.shape[0]}x{pool_y.shape[1]})')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print(f"原图尺寸: {img.shape}")
    print(f"卷积输出尺寸: {edge_x.shape}")
    print(f"池化输出尺寸: {pool_x.shape}")