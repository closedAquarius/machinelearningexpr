import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall

# 设置 Matplotlib 风格，使图表更美观
plt.style.use('ggplot')
import matplotlib as mpl

# --- 中文字体设置 ---
# 1. 检查系统中是否存在 SimHei 字体 (最常见的中文支持字体)
#    如果 SimHei 不存在，可能需要手动安装，或改为使用其他常见字体如 'Microsoft YaHei'
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
except ValueError:
    print("SimHei 字体未找到，尝试使用系统默认字体...")
    # 备用方案：使用系统默认的 sans-serif 字体
    pass

plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题
# --- 中文字体设置结束 ---


# ==============================================================================
# 模块 A: 数据加载与预处理
# ==============================================================================

print("="*50)
print("模块 A: 数据加载与预处理")
print("="*50)

# --- A.1 加载数据集 ---
print("--- A.1 加载 MNIST 数据集 ---")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
PIXELS = X_train.shape[1] * X_train.shape[2]  # 784
NUM_CLASSES = 10  # 10 个类别 (0-9)

# --- A.2 特征展平与类型转换 ---
print("--- A.2 特征展平与类型转换 ---")
# 展平图像数据 (N x 28 x 28 -> N x 784)
X_train = X_train.reshape(X_train.shape[0], PIXELS).astype('float32')
X_test = X_test.reshape(X_test.shape[0], PIXELS).astype('float32')

# --- A.3 特征归一化 ---
print("--- A.3 特征归一化 (0-1 范围) ---")
X_train /= 255.0
X_test /= 255.0

# --- A.4 标签独热编码 (One-Hot Encoding) ---
print("--- A.4 标签独热编码 ---")
Y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
Y_test = to_categorical(y_test, num_classes=NUM_CLASSES)


# ==============================================================================
# 模块 B: 模型构建与编译 (单层感知机)
# ==============================================================================

print("\n" + "="*50)
print("模块 B: 模型构建与编译")
print("="*50)

# --- B.1 模型创建与层添加 ---
print("--- B.1 创建序列模型并添加层 ---")
model = Sequential()
# 添加全连接输出层，同时集成 softmax 激活函数
model.add(Dense(units=NUM_CLASSES, input_shape=(PIXELS,), activation='softmax'))
model.summary()

# --- B.2 模型编译 ---
print("\n--- B.2 模型编译 ---")
sgd_optimizer = SGD()
custom_metrics = ['accuracy', Precision(), Recall()]

model.compile(
    loss='categorical_crossentropy',
    optimizer=sgd_optimizer,
    metrics=custom_metrics
)
print(f"模型编译完成。指标: {custom_metrics}")


# ==============================================================================
# 模块 C: 模型训练与评估
# ==============================================================================

print("\n" + "="*50)
print("模块 C: 模型训练与评估")
print("="*50)

# --- C.1 模型训练 (Fit) ---
print("--- C.1 开始训练模型 (200 Epochs) ---")
history = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=128,
    epochs=200,             # 训练 200 轮
    verbose=1,
    validation_split=0.2    # 20% 作为验证集
)
print("训练完成。")


# --- C.2 模型评估 (Evaluate) ---
print("\n--- C.2 模型评估 (测试集) ---")
score = model.evaluate(
    x=X_test,
    y=Y_test,
    verbose=1
)

# 打印评估结果
print("\n--- 评估结果 (测试集) ---")
print(f"总损失值 (Loss): {score[0]:.6f}")
print(f"准确率 (Accuracy): {score[1]:.4f}")
print(f"精确率 (Precision): {score[2]:.4f}")
print(f"召回率 (Recall): {score[3]:.4f}")


# ==============================================================================
# 模块 D: 可视化分析
# ==============================================================================

print("\n" + "="*50)
print("模块 D: 训练过程与结果可视化")
print("="*50)

hist_df = pd.DataFrame(history.history)
epochs_range = range(1, len(hist_df) + 1)

# --- D.1 损失和准确率曲线 ---
plt.figure(figsize=(14, 6))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs_range, hist_df['loss'], label='训练集损失', color='blue')
plt.plot(epochs_range, hist_df['val_loss'], label='验证集损失', color='orange')
plt.title('损失函数随训练次数变化 (Loss vs. Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, hist_df['accuracy'], label='训练集准确率', color='blue')
plt.plot(epochs_range, hist_df['val_accuracy'], label='验证集准确率', color='orange')
plt.title('准确率随训练次数变化 (Accuracy vs. Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- D.2 混淆矩阵热图 ---
print("\n--- D.2 混淆矩阵热图 ---")

# 1. 获取模型预测结果 (概率值)
Y_pred_prob = model.predict(X_test)
# 2. 将概率值转换为类别标签 (取概率最大的索引)
y_pred = np.argmax(Y_pred_prob, axis=1)

# 3. 计算混淆矩阵 (使用原始整数标签 y_test)
cm = confusion_matrix(y_test, y_pred)

# 4. 绘制热图
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,              # 显示每个单元格的数值
    fmt='d',                 # 确保数值显示为整数
    cmap='Blues',            # 使用蓝色系配色
    cbar=False               # 不显示颜色条
)
plt.title('测试集混淆矩阵热图')
plt.xlabel('预测标签 (Predicted Label)')
plt.ylabel('真实标签 (True Label)')
plt.show()

print("\n可视化分析完成！")

# 导入 t-SNE 模块
from sklearn.manifold import TSNE
import time

# ==============================================================================
# 模块 D (续): t-SNE 散点图
# ==============================================================================

print("\n--- D.3 t-SNE 散点图 (原始特征) ---")

# 由于 t-SNE 计算量较大，我们只对测试集中的一部分样本进行降维，
# 例如取前 2000 个样本，以提高计算速度。
N_SAMPLES = 2000
X_tsne = X_test[:N_SAMPLES]  # 原始特征 (784维)
y_tsne = y_test[:N_SAMPLES]  # 原始标签 (用于颜色标记)

print(f"正在对测试集中的前 {N_SAMPLES} 个样本执行 t-SNE 降维...")
time_start = time.time()

# 初始化 t-SNE 模型
# n_components=2: 降维到 2 维
# random_state=42: 确保结果可复现
tsne = TSNE(n_components=2, verbose=0, random_state=42)

# 执行降维
X_tsne_2d = tsne.fit_transform(X_tsne)

time_end = time.time()
print(f"t-SNE 降维完成，耗时: {time_end - time_start:.2f} 秒")

# --- 绘制散点图 ---
plt.figure(figsize=(10, 8))

# 遍历 10 个类别 (0 到 9)
for i in range(NUM_CLASSES):
    # 找到属于当前类别的所有样本的索引
    indices = y_tsne == i

    # 绘制散点图
    plt.scatter(
        X_tsne_2d[indices, 0],  # t-SNE 降维后的第一个维度
        X_tsne_2d[indices, 1],  # t-SNE 降维后的第二个维度
        label=str(i),  # 图例标签
        alpha=0.7,  # 透明度
        s=10  # 散点大小
    )

plt.title(f't-SNE 降维散点图 (原始 MNIST 特征, {N_SAMPLES} 样本)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='数字类别')
plt.grid(True)
plt.show()