import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall

# ==============================================================================
# 模块 A: 数据加载与预处理
# ==============================================================================

# --- A.1 加载数据集 ---
print("--- A.1 加载 MNIST 数据集 ---")
# 将数据集中的训练数据和测试数据分别赋值给对应的变量
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 查看初始维度
print(f"初始 X_train 维度: {X_train.shape}")
print(f"初始 y_train 维度: {y_train.shape}")

# --- A.2 特征展平与类型转换 ---
print("\n--- A.2 特征展平与类型转换 (float32) ---")
PIXELS = X_train.shape[1] * X_train.shape[2]  # 28 * 28 = 784

# 展平图像数据 (从 N x 28 x 28 变为 N x 784)
X_train = X_train.reshape(X_train.shape[0], PIXELS)
X_test = X_test.reshape(X_test.shape[0], PIXELS)

# 转换元素数据类型为 float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 验证展平后的形状和数据类型
print(f"展平后 X_train 形状: {X_train.shape}, 数据类型: {X_train.dtype}")

# --- A.3 特征归一化 ---
print("\n--- A.3 特征归一化 (0-1 范围) ---")
# 将像素值 (0-255) 归一化到 0.0-1.0 之间
X_train /= 255.0
X_test /= 255.0

# 检查归一化结果 (第 2 个例子的部分像素值)
print(f"归一化完成，检查 X_train[1][99:150]:\n{X_train[1][99:150]}")

# --- A.4 标签独热编码 (One-Hot Encoding) ---
print("\n--- A.4 标签独热编码 ---")
NUM_CLASSES = 10  # 类别总数 (0-9)

# 对训练集和测试集标签进行 One-Hot 编码
Y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
Y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

# 检查 One-Hot 编码结果 (Y_train 前五行)
print(f"Y_train 编码后的形状: {Y_train.shape}")
print(f"Y_train 前五行:\n{Y_train[:5]}")


# ==============================================================================
# 模块 B: 模型构建与编译 (单层感知机)
# ==============================================================================

print("\n" + "="*50)
print("模块 B: 模型构建与编译 (单层感知机)")
print("="*50)

# --- B.1 模型创建与层添加 ---
print("--- B.1 创建序列模型并添加层 ---")
model = Sequential()

# 添加全连接层 (Dense)，同时指定激活函数 'softmax'
# units=10: 输出维度为 10 (对应 10 个类别)
# input_shape=(784,): 指定输入数据的形状
model.add(Dense(units=10, input_shape=(PIXELS,), activation='softmax'))

# 通过 summary 方法查看模型结构
model.summary()

# --- B.2 模型编译 ---
print("\n--- B.2 模型编译 ---")
sgd_optimizer = SGD()

# 定义评估指标列表：损失、准确率、精确率、召回率
custom_metrics = ['accuracy', Precision(), Recall()]

model.compile(
    loss='categorical_crossentropy',  # 损失函数 (多分类)
    optimizer=sgd_optimizer,         # 优化器
    metrics=custom_metrics           # 性能评估指标
)
print(f"模型编译完成。指标: {custom_metrics}")


# ==============================================================================
# 模块 C: 模型训练与评估 (单层感知机)
# ==============================================================================

print("\n" + "="*50)
print("模块 C: 模型训练与评估")
print("="*50)

# --- C.1 模型训练 (Fit) ---
print("--- C.1 开始训练模型 (200 Epochs) ---")
history = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=128,         # 批量大小
    epochs=200,             # 训练轮次
    verbose=1,              # 显示训练进度条
    validation_split=0.2    # 20% 训练集作为验证集
)
print("训练完成。")


# --- C.2 模型评估 (Evaluate) ---
print("\n--- C.2 模型评估 (测试集) ---")
# 在全新的测试集上评估模型性能
score = model.evaluate(
    x=X_test,
    y=Y_test,
    verbose=1
)

# 打印评估结果 (损失和准确率)
print("\n--- 评估结果 ---")
print(f"总损失值 (Loss): {score[0]:.6f}")
print(f"准确率 (Accuracy): {score[1]:.4f}")
print(f"精确率 (Precision): {score[2]:.4f}")
print(f"召回率 (Recall): {score[3]:.4f}")