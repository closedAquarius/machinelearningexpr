import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
import matplotlib

# 强制使用 Agg 后端，避免图形显示，专注于文件输出
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# --- t-SNE 依赖 ---
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import time

# --- Keras/TensorFlow 相关的文本处理工具 (用于数据预处理，保持一致性) ---
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- 文件路径 ---
TRAIN_FILE_PATH = './review/drugsComTrain_raw.csv'
TEST_FILE_PATH = './review/drugsComTest_raw.csv'
OUTPUT_DIR = 'output2.0torchplus'

# --- 确保输出目录存在 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# --- 1. 设备设置 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

# --- 中文字体设置 (可选) ---
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


# --- 辅助函数 ---
def map_rating_to_sentiment(rating):
    """映射评分到 PyTorch/Numpy 索引 (0, 1, 2)"""
    if rating <= 4:
        return 0  # 消极 (Negative)
    elif rating <= 6:
        return 1  # 中性 (Neutral)
    else:
        return 2  # 积极 (Positive)


def calculate_accuracy(y_pred, y_true):
    """计算模型准确率"""
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).sum().item()
    return correct_predictions / y_true.size(0)


# --- 绘图函数 (修改为保存文件) ---
def plot_history(history):
    f, ax = plt.subplots(1, 2, figsize=(16, 7))

    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']

    epochs = range(1, len(acc) + 1)

    # --- 准确率图 ---
    plt.sca(ax[0])
    ax[0].xaxis.set_major_locator(MultipleLocator(1))
    plt.plot(epochs, acc, marker='o', label='Training acc')
    plt.plot(epochs, val_acc, marker='o', label='Validation acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.title('Training and validation accuracy')
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # --- 损失图 ---
    plt.sca(ax[1])
    ax[1].xaxis.set_major_locator(MultipleLocator(1))
    plt.plot(epochs, loss, marker='o', label='Training loss')
    plt.plot(epochs, val_loss, marker='o', label='Validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('Training and validation loss')
    plt.grid(axis='y', linestyle='--')
    plt.legend()

    # ❗ 保存文件，而不是显示
    save_path = os.path.join(OUTPUT_DIR, '01_training_history.png')
    plt.savefig(save_path)
    plt.close(f)
    print(f"✅ 训练历史图已保存到: {save_path}")


# --- t-SNE 可视化函数 (已修复兼容性问题) ---
def visualize_tsne(features, labels, title, filename):
    """
    使用 t-SNE 将高维特征降维到 2D 并绘图。
    已修复旧版本 scikit-learn 中 n_iter 参数导致的 TypeError。
    """
    print(f"正在对 {len(features)} 个样本进行 t-SNE 降维...")

    # 由于 t-SNE 计算量大，如果样本数超过 5000，考虑使用 PCA 初步降维
    if features.shape[0] > 5000:
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(features)
        print("已使用 PCA 降维至 50 维。")
        # ❗ 修复：将 n_iter 替换为 max_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        features_2d = tsne.fit_transform(features_pca)
    else:
        # ❗ 修复：将 n_iter 替换为 max_iter
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
    print(f"✅ {title} 图已保存到: {save_path}")


# ----------------------------------------
# I. 数据加载与预处理 (保持不变)
# ----------------------------------------

print("\n--- I. 数据加载与 Keras 预处理 ---")

# 1. 读取原始数据
try:
    train_data = pd.read_csv(TRAIN_FILE_PATH)
    test_data = pd.read_csv(TEST_FILE_PATH)

    # 清理缺失值，保证数据完整性
    train_data = train_data.dropna(subset=['rating', 'review']).reset_index(drop=True)
    test_data = test_data.dropna(subset=['rating', 'review']).reset_index(drop=True)

except FileNotFoundError:
    print(f"错误: 原始数据文件未找到，请检查路径: {TRAIN_FILE_PATH}")
    exit()

# 2. 提取评论和标签 (PyTorch 使用非独热编码标签)
train_review = train_data['review'].astype(str)
test_review = test_data['review'].astype(str)
train_label = train_data['rating'].apply(map_rating_to_sentiment).values  # NumPy 数组
test_label = test_data['rating'].apply(map_rating_to_sentiment).values  # NumPy 数组

# 3. Tokenizer 拟合和序列化
tokenizer = Tokenizer(num_words=WORDS)
tokenizer.fit_on_texts(train_review)
train_sequence = tokenizer.texts_to_sequences(train_review)
test_sequence = tokenizer.texts_to_sequences(test_review)

# 4. Padding
x_train = pad_sequences(train_sequence, maxlen=LENGTH)
x_test = pad_sequences(test_sequence, maxlen=LENGTH)

print(f"数据预处理完成。x_train 形状: {x_train.shape}")

# ----------------------------------------
# II. PyTorch Dataset & DataLoader (保持不变)
# ----------------------------------------

print("\n--- II. PyTorch Dataset & DataLoader ---")


class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


train_dataset = ReviewDataset(x_train, train_label)
test_dataset = ReviewDataset(x_test, test_label)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# ❗ 使用一个较小的 DataLoader 来提取特征，以加速 t-SNE 计算
subset_indices = np.random.choice(len(test_dataset), size=min(5000, len(test_dataset)), replace=False)
test_subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
feature_loader = DataLoader(test_subset_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 用于完整评估

# ----------------------------------------
# III. PyTorch 模型定义 (nn.Module)
# ----------------------------------------

print("\n--- III. PyTorch 模型定义 (新增特征提取方法) ---")


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # 仅用于训练和评估，返回 logits
        embedded = self.embedding(text)
        # LSTM 返回 (output, (h_n, c_n))
        _, (hidden, _) = self.lstm(embedded)
        # hidden 是 (1, Batch_size, Hidden_dim)，我们取 h_n，然后移除维度 0
        hidden = hidden.squeeze(0)
        return self.fc(hidden)

    def extract_features(self, text, layer_name='lstm'):
        # 用于特征可视化，返回特定层的输出
        embedded = self.embedding(text)

        if layer_name == 'embedding':
            # 返回序列中所有词的平均嵌入向量 (用于词嵌入层的特征)
            return torch.mean(embedded, dim=1)

            # LSTM 特征 (最后一个时间步的隐藏状态)
        _, (hidden, _) = self.lstm(embedded)
        features = hidden.squeeze(0)  # (batch_size, hidden_dim)

        return features


# 实例化模型并移动到 GPU
model = SentimentLSTM(WORDS, DEPTH, DEPTH, NUM_CLASSES).to(device)

# ----------------------------------------
# IV. 优化器与损失函数 (保持不变)
# ----------------------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------------------
# V. 训练与评估函数 (保持不变)
# ----------------------------------------

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc, total_count = 0, 0, 0

    for sequences, labels in loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        acc = calculate_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        total_acc += acc * len(labels)
        total_count += len(labels)
    return total_loss / total_count, total_acc / total_count


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc, total_count = 0, 0, 0
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)
            total_loss += loss.item() * len(labels)
            total_acc += acc * len(labels)
            total_count += len(labels)
    return total_loss / total_count, total_acc / total_count


# ----------------------------------------
# VI. 训练主循环 (保持不变)
# ----------------------------------------

print(f"\n--- 7. 模型训练 ({EPOCHS} Epochs, 使用 {device}) ---")

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion, device)

    history['loss'].append(train_loss)
    history['acc'].append(train_acc)
    history['val_loss'].append(valid_loss)
    history['val_acc'].append(valid_acc)

    print(f'Epoch: {epoch + 1:02} | '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | '
          f'Test Loss: {valid_loss:.4f} | Test Acc: {valid_acc * 100:.2f}%')

print("✅ PyTorch 模型训练完成。")

# ----------------------------------------
# VII. 特征提取与 t-SNE 可视化 (已修复，保存到文件)
# ----------------------------------------

print("\n--- 10. 特征提取与 t-SNE 可视化 ---")


def get_features_and_labels(model, loader, device, layer_name):
    """从指定层提取特征和真实标签"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)

            # 使用模型新增的特征提取方法
            features = model.extract_features(sequences, layer_name)

            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


# --- 提取 LSTM 输出特征 (用于分类的特征) ---
lstm_features, lstm_labels = get_features_and_labels(model, feature_loader, device, 'lstm')
visualize_tsne(
    lstm_features,
    lstm_labels,
    f'LSTM 输出特征 t-SNE 可视化 (用于分类, 样本数: {len(lstm_features)})',
    '02_lstm_tsne_features.png'
)

# ----------------------------------------
# VIII. 最终评估和绘图 (保存到文件)
# ----------------------------------------

print("\n--- 8. 模型最终评估 ---")
final_loss, final_acc = evaluate(model, test_loader, criterion, device)

print(f"\n--- 评估结果 ---")
print(f"最终测试集损失 (Test Loss): {final_loss:.4f}")
print(f"最终测试集准确率 (Test Accuracy): {final_acc:.4f}")

print("\n--- 9. 训练历史可视化 ---")
plot_history(history)