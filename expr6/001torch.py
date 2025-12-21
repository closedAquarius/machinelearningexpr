import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd
import matplotlib
# 强制使用 Tkinter 后端，这是一个常见的外部 GUI 后端
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# --- Keras/TensorFlow 相关的文本处理工具 (用于数据预处理，保持一致性) ---
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- 文件路径 ---
TRAIN_FILE_PATH = './review/drugsComTrain_raw.csv'
TEST_FILE_PATH = './review/drugsComTest_raw.csv'

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
        return 0  # 消极
    elif rating <= 6:
        return 1  # 中性
    else:
        return 2  # 积极


def calculate_accuracy(y_pred, y_true):
    """计算模型准确率"""
    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).sum().item()
    return correct_predictions / y_true.size(0)


# --- 绘图函数 (沿用 Keras 风格) ---
def plot_history(history):
    f, ax = plt.subplots(1, 2, figsize=(16, 7))

    # 兼容 Keras 和 PyTorch 的 key
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

    plt.show()


# ----------------------------------------
# I. 数据加载与预处理 (Keras/TF 逻辑复用)
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
# II. PyTorch Dataset & DataLoader
# ----------------------------------------

print("\n--- II. PyTorch Dataset & DataLoader ---")


class ReviewDataset(Dataset):
    def __init__(self, sequences, labels):
        # 转换为 PyTorch Tensor
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


train_dataset = ReviewDataset(x_train, train_label)
test_dataset = ReviewDataset(x_test, test_label)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------
# III. PyTorch 模型定义 (nn.Module)
# ----------------------------------------

print("\n--- III. PyTorch 模型定义 ---")


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=True 保持形状与 Keras 一致
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        # 仅使用最后一个时间步的隐藏状态进行分类
        _, (hidden, _) = self.lstm(embedded)
        # hidden 形状: (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)


# 实例化模型并移动到 GPU
model = SentimentLSTM(WORDS, DEPTH, DEPTH, NUM_CLASSES).to(device)

# ----------------------------------------
# IV. 优化器与损失函数
# ----------------------------------------

criterion = nn.CrossEntropyLoss()
# 使用 Adam 优化器作为 Keras RMSprop 的高性能替代品
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------------------------------
# V. 训练与评估函数
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
# VI. 训练主循环 (3.7.3.5 训练及评估)
# ----------------------------------------

print(f"\n--- 7. 模型训练 ({EPOCHS} Epochs, 使用 {device}) ---")

history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    # 使用 test_loader 作为验证集进行评估
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
# VII. 评估和绘图
# ----------------------------------------

print("\n--- 8. 模型最终评估 ---")
final_loss, final_acc = evaluate(model, test_loader, criterion, device)

print(f"\n--- 评估结果 ---")
print(f"最终测试集损失 (Test Loss): {final_loss:.4f}")
print(f"最终测试集准确率 (Test Accuracy): {final_acc:.4f}")

print("\n--- 9. 训练历史可视化 ---")
plot_history(history)