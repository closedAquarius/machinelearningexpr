import pandas as pd
import matplotlib

# 强制使用 Agg 后端，专注于文件保存
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
import numpy as np

# --- Keras/TensorFlow 相关的文本处理工具 ---
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.metrics import Recall, Precision

# --- Keras/TensorFlow 模型构建模块 ---
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --- t-SNE 依赖 ---
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# --- 文件路径 ---
TRAIN_FILE_PATH = './review/drugsComTrain_raw.csv'
TEST_FILE_PATH = './review/drugsComTest_raw.csv'
OUTPUT_DIR = 'output2.0plus'  # ❗ 新增输出目录

# --- 确保输出目录存在 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# --- 中文字体设置 (可选，确保中文显示正常) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# --- 超参数定义 ---
WORDS = 60000  # 词汇表大小
LENGTH = 200  # 序列最大长度
DEPTH = 128  # 嵌入维度和 LSTM 单元数
NUM_CLASSES = 3  # 类别数
BATCH_SIZE = 64
EPOCHS = 10


def map_rating_to_sentiment(rating):
    """
    将 1-10 的评分映射到 -1, 0, 1 三个情感类别:
    -1: Negative (1-4)
    0: Neutral (5-6)
    1: Positive (7-10)
    """
    if rating <= 4:
        return -1  # 消极
    elif rating <= 6:
        return 0  # 中性
    else:
        return 1  # 积极


def generate_sentiment_labels(file_path):
    """加载数据并生成情感标签"""
    try:
        data = pd.read_csv(file_path)
        data = data.dropna(subset=['rating', 'review']).reset_index(drop=True)
        rating = data['rating']
        review = data['review'].astype(str)
        label = rating.apply(map_rating_to_sentiment)
        return review, label, data
    except FileNotFoundError:
        print(f"错误: 文件未找到，请检查路径: {file_path}")
        return None, None, None


# ----------------------------------------
# I. 数据加载与预处理
# ----------------------------------------

print("--- I. 数据加载与 Keras 预处理 ---")

# (1) 读取原始数据并生成标签
train_review, train_label, train_data = generate_sentiment_labels(TRAIN_FILE_PATH)
test_review, test_label, test_data = generate_sentiment_labels(TEST_FILE_PATH)

if train_review is None or test_review is None:
    exit()

print(f"训练集数据量: {len(train_data)}")
print(f"测试集数据量: {len(test_data)}")


# ----------------------------------------
# II. 标签分布绘图 (修改为保存文件)
# ----------------------------------------

def plot_sentiment_pie(train_label, test_label, output_dir):
    """绘制训练集和测试集的情感标签扇形图并保存"""

    train_counts = train_label.value_counts().sort_index()
    test_counts = test_label.value_counts().sort_index()

    label_names = ['消极 (-1)', '中性 (0)', '积极 (1)']
    # 确保索引 [0, 1, 2] 对应到颜色
    colors_map = {-1: '#ff9999', 0: '#ffcc99', 1: '#99ff99'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('处理后的情感标签分布 (Rating -> 3 Classes)', fontsize=16)

    # 训练集
    axes[0].pie(
        train_counts,
        labels=[label_names[l + 1] for l in train_counts.index],  # 映射 -1->0, 0->1, 1->2
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors_map[l] for l in train_counts.index]
    )
    axes[0].set_title(f'训练集 ({train_label.shape[0]} 样本)', fontsize=14)
    axes[0].axis('equal')

    # 测试集
    axes[1].pie(
        test_counts,
        labels=[label_names[l + 1] for l in test_counts.index],
        autopct='%1.1f%%',
        startangle=90,
        colors=[colors_map[l] for l in test_counts.index]
    )
    axes[1].set_title(f'测试集 ({test_label.shape[0]} 样本)', fontsize=14)
    axes[1].axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, '00_sentiment_label_distribution.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f"\n✅ 情感标签扇形图已保存到: {save_path}")


plot_sentiment_pie(train_label, test_label, OUTPUT_DIR)

# ----------------------------------------
# III. 文本序列化与标签独热编码
# ----------------------------------------

print("\n--- III. 文本序列化与标签独热编码 ---")
# 1. Tokenizer 拟合和序列化
tokenizer = Tokenizer(num_words=WORDS)
tokenizer.fit_on_texts(train_review)
train_sequence = tokenizer.texts_to_sequences(train_review)
test_sequence = tokenizer.texts_to_sequences(test_review)

# 2. Padding
x_train = pad_sequences(train_sequence, maxlen=LENGTH)
x_test = pad_sequences(test_sequence, maxlen=LENGTH)

# 3. 标签独热编码
train_label_mapped = train_label + 1  # [-1, 0, 1] -> [0, 1, 2]
test_label_mapped = test_label + 1
y_train = to_categorical(train_label_mapped, num_classes=NUM_CLASSES)
y_test = to_categorical(test_label_mapped, num_classes=NUM_CLASSES)

print(f"x_train 形状: {x_train.shape}")
print(f"y_train 形状: {y_train.shape}")

# ----------------------------------------
# IV. 模型搭建与编译
# ----------------------------------------

print("\n--- IV. 模型搭建与编译 ---")
# (1) 初始化 Sequential 模型
model = Sequential()
model.add(Embedding(WORDS, DEPTH, input_length=LENGTH, name='embedding_layer'))
model.add(LSTM(DEPTH, name='lstm_layer'))
model.add(Dense(NUM_CLASSES, activation='softmax', name='output_layer'))

model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc', Recall(), Precision()]
)
model.summary()

# ----------------------------------------
# V. 训练模型
# ----------------------------------------
print(f"\n--- V. 模型训练 ({EPOCHS} Epochs) ---")
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    validation_split=0.25
)
print("✅ 模型训练完成。")


# ----------------------------------------
# VI. 训练历史绘图 (修改为保存文件)
# ----------------------------------------
def plot_history(history, output_dir):
    """绘制训练集和验证集的准确率及损失曲线并保存"""
    f, ax = plt.subplots(1, 2, figsize=(16, 7))

    acc = history.history.get('acc', history.history.get('accuracy'))
    val_acc = history.history.get('val_acc', history.history.get('val_accuracy'))
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # 准确率图
    plt.sca(ax[0])
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    plt.plot(epochs, acc, marker='o', label='Training acc')
    plt.plot(epochs, val_acc, marker='o', label='Validation acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training and validation accuracy')
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # 损失图
    plt.sca(ax[1])
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    plt.plot(epochs, loss, marker='o', label='Training loss')
    plt.plot(epochs, val_loss, marker='o', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training and validation loss')
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    save_path = os.path.join(output_dir, '01_training_history.png')
    plt.savefig(save_path)
    plt.close(f)
    print(f"\n✅ 训练历史图已保存到: {save_path}")


plot_history(history, OUTPUT_DIR)

# ----------------------------------------
# VII. 特征提取与 t-SNE 可视化 (新增)
# ----------------------------------------

print("\n--- VII. 特征提取与 t-SNE 可视化 ---")


def extract_features(model, layer_name, data_x):
    """
    创建一个新的 Keras 模型，用于提取指定层的输出。
    """
    # 获取指定层的输出张量
    layer_output = model.get_layer(layer_name).output

    # 构建特征提取模型
    feature_model = Model(inputs=model.input, outputs=layer_output)

    # 预测并提取特征
    features = feature_model.predict(data_x, batch_size=BATCH_SIZE, verbose=0)
    return features


# --- 修正后的：t-SNE 可视化函数 ---
def visualize_tsne(features, labels, title, filename):
    """
    使用 t-SNE 将高维特征降维到 2D 并绘图。
    """
    print(f"正在对 {len(features)} 个样本进行 t-SNE 降维...")

    # 由于 t-SNE 计算量大，如果样本数超过 5000，考虑使用 PCA 初步降维
    if features.shape[0] > 5000:
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(features)
        print("已使用 PCA 降维至 50 维。")
        # ❗ 将 n_iter 替换为 max_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = tsne.fit_transform(features_pca)
    else:
        # ❗ 将 n_iter 替换为 max_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    # 定义类别和颜色
    colors = ['#ff4d4d', '#ffcc00', '#00cc66']  # 消极 (红), 中性 (黄), 积极 (绿)
    labels_map = {0: '消极 (Negative)', 1: '中性 (Neutral)', 2: '积极 (Positive)'}

    for i in range(NUM_CLASSES):
        # 提取属于当前类别的样本
        indices = np.where(labels == i)
        plt.scatter(
            features_2d[indices, 0],
            features_2d[indices, 1],
            c=colors[i],
            label=labels_map[i],
            alpha=0.6,
            s=15
        )

    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # ❗ 保存文件，而不是显示
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ t-SNE 可视化图已保存到: {save_path}")


# --- 提取 LSTM 输出特征 (用于分类的特征) ---
# 使用测试集数据进行提取
lstm_features = extract_features(model, 'lstm_layer', x_test)
visualize_tsne(
    lstm_features,
    y_test,
    f'LSTM 输出特征 t-SNE 可视化 (用于分类, 样本数: {min(5000, x_test.shape[0])})',
    '02_lstm_tsne_features.png',
    OUTPUT_DIR
)

# ----------------------------------------
# VIII. 模型评估
# ----------------------------------------

print("\n--- VIII. 模型评估 ---")

test_results = model.evaluate(x_test, y_test, verbose=1)

# 根据 model.compile 中的 metrics 顺序获取结果
test_loss = test_results[0]
test_acc = test_results[1]

print(f"\n--- 评估结果 ---")
print(f"测试集损失 (Test Loss): {test_loss:.4f}")
print(f"测试集准确率 (Test Accuracy): {test_acc:.4f}")