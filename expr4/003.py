import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix # 用于混淆矩阵
from sklearn.manifold import TSNE # 用于 t-SNE 降维
import time # 用于 t-SNE 计时

# 设置 Matplotlib 风格 (可选)
plt.style.use('ggplot')

# ==============================================================================
# 🛠️ 修复中文乱码问题的代码 🛠️
# 解决绘图时中文标签显示为方块的问题
# ==============================================================================
try:
    # 尝试设置支持中文的字体 SimHei
    plt.rcParams['font.sans-serif'] = ['SimHei']
except:
    # 如果 SimHei 不可用，使用系统默认 sans-serif 字体（如 Arial Unicode MS）
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
# ==============================================================================


# ==============================================================================
# 模块 A: 数据加载与预处理
# ==============================================================================

# --- A.1 加载数据集 ---
print("--- A.1 加载 Fashion-MNIST 数据集 ---")
(X_train, y_train), (X_test, y_test) = mnist.load_data() # **加载 Fashion-MNIST**

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

# --- A.3 特征归一化 ---
print("\n--- A.3 特征归一化 (0-1 范围) ---")
# 将像素值 (0-255) 归一化到 0.0-1.0 之间
X_train /= 255.0
X_test /= 255.0

# --- A.4 标签独热编码 (One-Hot Encoding) ---
print("\n--- A.4 标签独热编码 ---")
NUM_CLASSES = 10  # 类别总数 (0-9)

# 对训练集和测试集标签进行 One-Hot 编码
Y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
Y_test = to_categorical(y_test, num_classes=NUM_CLASSES)


# ==============================================================================
# 模块 B: 模型构建与编译 (三层全连接网络)
# ==============================================================================

print("\n" + "="*50); print("模块 B: 模型构建与编译 (三层全连接网络)"); print("="*50)

print("--- 1. 创建序列模型对象 ---")
model = Sequential()

print("\n--- 2. 添加两个隐藏层和输出层 ---")

# **第一层 (隐藏层 1)**: 输入层 (784) -> 隐藏层 (128, ReLU)
model.add(Dense(
    units=128,
    input_shape=(784,), # 指定第一层的输入形状
    activation='relu',
    name='hidden_layer_1' # 添加名称，便于后续提取特征
))

# **第二层 (隐藏层 2)**: 隐藏层 (128) -> 隐藏层 (128, ReLU)
model.add(Dense(
    units=128,
    activation='relu',
    name='hidden_layer_2'
))

# **第三层 (输出层)**: 隐藏层 (128) -> 输出层 (10, Softmax)
model.add(Dense(
    units=10,
    activation='softmax',
    name='output_layer'
))

print("三个全连接层添加成功。")


print("\n--- 3. 查看的模型摘要 (Summary) ---")
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
print("模型编译完成。")


# ==============================================================================
# 模块 C: 模型训练与评估
# ==============================================================================

print("\n" + "="*50)
print("模块 C: 模型训练与评估")
print("="*50)

# --- C.1 模型训练 (Fit) ---
print("--- C.1 开始训练模型 (20 Epochs) ---")
# 减少训练轮次到 20 以便快速演示新模型效果
history = model.fit(
    x=X_train,
    y=Y_train,
    batch_size=128,
    epochs=20, # 训练轮次
    verbose=1,
    validation_split=0.2
)
print("训练完成。")


# --- C.2 模型评估 (Evaluate) ---
print("\n--- C.2 模型评估 (测试集) ---")
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


# ==============================================================================
# 模块 D: 训练过程与结果可视化
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
plt.ylabel('损失')
plt.legend()
plt.grid(True)

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs_range, hist_df['accuracy'], label='训练集准确率', color='blue')
plt.plot(epochs_range, hist_df['val_accuracy'], label='验证集准确率', color='orange')
plt.title('准确率随训练次数变化 (Accuracy vs. Epochs)')
plt.xlabel('Epoch')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() #


# --- D.2 混淆矩阵热图 ---
print("\n--- D.2 混淆矩阵热图 ---")

Y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(Y_pred_prob, axis=1)

# 计算混淆矩阵 (使用原始整数标签 y_test)
cm = confusion_matrix(y_test, y_pred)

# 绘制热图
plt.figure(figsize=(8, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False
)
plt.title('测试集混淆矩阵热图')
plt.xlabel('预测标签 (Predicted Label)')
plt.ylabel('真实标签 (True Label)')
plt.show() #


# --- D.3 t-SNE 散点图 (模型特征) ---
print("\n--- D.3 t-SNE 散点图 (模型提取特征) ---")

N_SAMPLES = 2000

# **关键变化：提取第一个隐藏层的输出特征**
# 实例化一个新的模型，只包含输入层到第一个隐藏层
feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer('hidden_layer_1').output
)

print(f"正在提取第一个隐藏层 (128维) 的特征...")
X_features = feature_extractor.predict(X_test[:N_SAMPLES], verbose=0)
y_tsne = y_test[:N_SAMPLES]

print(f"正在对特征集中的前 {N_SAMPLES} 个样本执行 t-SNE 降维 (128维 -> 2维)...")
time_start = time.time()

# 初始化 t-SNE 模型
tsne = TSNE(n_components=2, verbose=0, random_state=42)

# 执行降维
X_tsne_2d = tsne.fit_transform(X_features)

time_end = time.time()
print(f"t-SNE 降维完成，耗时: {time_end - time_start:.2f} 秒")

# --- 绘制散点图 ---
plt.figure(figsize=(10, 8))

# 遍历 10 个类别 (0 到 9)
for i in range(NUM_CLASSES):
    indices = y_tsne == i

    # 绘制散点图
    plt.scatter(
        X_tsne_2d[indices, 0],
        X_tsne_2d[indices, 1],
        label=str(i),
        alpha=0.7,
        s=10
    )

plt.title(f't-SNE 降维散点图 (Hidden Layer 1 特征, {N_SAMPLES} 样本)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='服装类别')
plt.grid(True)
plt.show() #

print("\n可视化分析完成！")