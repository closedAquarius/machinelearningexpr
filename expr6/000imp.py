import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
from tqdm import tqdm

# ❗ Matplotlib 后端设置：防止在某些IDE/环境（如PyCharm）中出现绘图后端错误
import matplotlib

matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image

# --- 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
except ValueError:
    print("SimHei 字体未找到，尝试使用系统默认字体...")
    pass
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题
# --- 中文字体设置结束 ---

# ======================================================================
#                       I. 超参数与配置
# ======================================================================

# 训练时批次大小
TRAIN_BATCH_SIZE = 32
# 测试时批次大小 (推荐设置为总样本数 66)
TEST_BATCH_SIZE = 66
# 总共训练的轮数
EPOCHS = 60
# 加噪率 (从 0.2 降至 0.1，减轻平滑)
AE_NOISE_STD = 0.1  # ✅ 改进点 1: 降低 AE 噪声强度
# 权重衰减 (L2 正则化)
WEIGHT_DECAY = 1e-4  # ✅ 改进点 2: 引入 L2 正则化
# 学习率
LR = 1e-3
# 数据路径
DATA_PATH = 'covid19'
# 模型保存路径
AE_MODEL_PATH = './autoencoder_model_imp.pth'
CNN_MODEL_PATH = './cnn_model_imp.pth'
# 可视化输出目录 (新增)
OUTPUT_DIR = 'output1.0_imp'

# 优先使用GPU训练 (cuda)，如果不可用则使用CPU。
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Covid', 'Normal', 'Viral Pneumonia']

print(f"使用的设备: {DEVICE}")
print(f"训练批次大小: {TRAIN_BATCH_SIZE}")
print(f"总训练轮数: {EPOCHS}")

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: {OUTPUT_DIR}")


# ======================================================================
#                      II. 数据处理与工具函数
# ======================================================================

def add_gaussian_noise(images, std_dev=AE_NOISE_STD):
    """为图像张量添加高斯噪声"""
    # ✅ 改进点 1: 使用降低后的 AE_NOISE_STD
    noise = torch.randn_like(images) * std_dev
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images


def get_dataloaders(data_path):
    """定义 Transforms 并加载数据集"""
    print("\n--- 读取数据与定义 DataLoader ---")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        root=f'{data_path}/train',
        transform=transform
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    test_dataset = datasets.ImageFolder(
        root=f'{data_path}/noisy_test',
        transform=transform
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False
    )

    print(f"训练集样本总数: {len(train_dataset)}")
    print(f"测试集样本总数: {len(test_dataset)}")

    return train_loader, test_loader


# ======================================================================
#                        III. 模型架构定义
# ======================================================================

# --- 1. 自编码器 (Autoencoder) ---
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# --- 2. 卷积神经网络 (CNN) ---
class CNN(nn.Module):
    def __init__(self, in_channels=1):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # 展平后的输入尺寸: 32 channels * 16 * 16 = 8192

        # ✅ 改进点 4: 减小 FC 层尺寸 (128 -> 64, 32 -> 16)
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=len(CLASS_NAMES))

        # ✅ 改进点 3: 引入 Dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # x.view(-1, 8192) 要求 x 的总元素数量能被 8192 整除
        x = x.view(-1, 32 * 16 * 16)

        self.features = x  # 用于 t-SNE

        # ✅ 改进点 3: 在全连接层应用 Dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ======================================================================
#                           IV. 训练与测试函数
# ======================================================================

def train_autoencoder_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """单轮自编码器训练"""
    model.train()
    running_loss = 0.0

    for data, _ in tqdm(dataloader, desc=f"AE Train Epoch {epoch}"):
        data = data.to(device)
        noisy_data = add_gaussian_noise(data)

        optimizer.zero_grad()
        output = model(noisy_data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"AE Train - Epoch {epoch}: Loss: {avg_loss:.6f}")
    return avg_loss


def train_cnn_epoch(cnn_model, ae_model, dataloader, criterion, optimizer, device, epoch):
    """单轮 CNN 训练 (带去噪)"""
    cnn_model.train()
    ae_model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data, target in tqdm(dataloader, desc=f"CNN Train Epoch {epoch}"):
        data, target = data.to(device), target.to(device)

        noisy_data = add_gaussian_noise(data)

        with torch.no_grad():
            denoised_data = ae_model(noisy_data)

        optimizer.zero_grad()
        output = cnn_model(denoised_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        total_samples += target.size(0)
        correct_predictions += (predicted == target).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    print(f"CNN Train - Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def test_cnn_epoch(cnn_model, ae_model, dataloader, criterion, device, epoch):
    """单轮 CNN 测试，并计算多个指标"""
    cnn_model.eval()
    ae_model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc=f"CNN Test Epoch {epoch}"):
            data, target = data.to(device), target.to(device)

            denoised_data = ae_model(data)

            output = cnn_model(denoised_data)

            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples

    report = classification_report(
        all_targets,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics = report['macro avg']

    print(f"CNN Test - Epoch {epoch}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    print(
        f"  P/R/F1 (Macro Avg): P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    return avg_loss, accuracy, metrics['precision'], metrics['recall'], metrics[
        'f1-score'], all_targets, all_predictions


# ======================================================================
#                           V. 可视化函数 (已修改为保存到 output1.0 目录)
# ======================================================================

def plot_metrics(history, epochs, output_dir):
    """绘制 Loss, Accuracy, Precision, Recall, F1-Score 曲线图并保存"""
    print("\n--- 绘制和保存指标曲线图 ---")
    if not history or not history[0]:
        print("无历史数据，跳过指标可视化。")
        return

    epochs_range = list(range(1, epochs + 1))
    train_loss = history[0];
    test_loss = history[1]
    train_acc = history[2];
    test_acc = history[3]
    test_precision = history[4];
    test_recall = history[5];
    test_f1 = history[6]

    # --- 图 1: Loss 和 Accuracy ---
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs_range, test_loss, label='Test Loss', marker='o', linestyle='--')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend()
    plt.grid(axis='y', linestyle='--');
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_acc, label='Train Accuracy', marker='o', linestyle='-')
    plt.plot(epochs_range, test_acc, label='Test Accuracy', marker='o', linestyle='--')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs');
    plt.ylabel('Accuracy (%)');
    plt.legend()
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--');
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_loss_acc.png'))
    plt.close()  # 关闭图表，释放内存

    # --- 图 2: Precision, Recall, F1-Score ---
    plt.figure(figsize=(6, 5))
    plt.plot(epochs_range, test_precision, label='Precision (Macro Avg)', marker='s', linestyle='-')
    plt.plot(epochs_range, test_recall, label='Recall (Macro Avg)', marker='^', linestyle='-')
    plt.plot(epochs_range, test_f1, label='F1-Score (Macro Avg)', marker='d', linestyle='-')

    plt.title('P/R/F1-Score vs. Epochs (Test Set)')
    plt.xlabel('Epochs');
    plt.ylabel('Score');
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--');
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_prf1.png'))
    plt.close()


def visualize_denoising(model, dataloader, device, output_dir, n_display=10):
    """绘制自编码器去噪结果：输入和输出对比图并保存"""
    print("\n--- 绘制和保存自编码器去噪结果 ---")
    model.eval()
    tp = transforms.ToPILImage()

    try:
        data, _ = next(iter(dataloader))
    except StopIteration:
        print("DataLoader为空，无法获取批次。")
        return

    data = data.to(device)

    with torch.no_grad():
        output = model(data)

    plt.figure(figsize=(20, 4))
    plt.suptitle(f"Autoencoder Denoising Result (First {n_display} Samples)", fontsize=16)

    for i in range(min(n_display, data.size(0))):
        # 绘制加噪输入图片（第一行）
        ax_noisy = plt.subplot(2, n_display, i + 1)
        noisy_img = tp(data[i].cpu())
        plt.imshow(noisy_img, cmap='gray')
        if i == 0: ax_noisy.set_title("Noisy Input", fontsize=10)
        plt.axis('off')

        # 绘制去噪后的输出图片（第二行）
        ax_reconstructed = plt.subplot(2, n_display, i + n_display + 1)
        reconstructed = tp(output[i].cpu())
        plt.imshow(reconstructed, cmap='gray')
        if i == 0: ax_reconstructed.set_title("Denoised Output", fontsize=10)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'autoencoder_denoising.png'))
    plt.close()


def plot_tsne_features(cnn_model, ae_model, dataloader, device, all_targets, all_predictions, output_dir):
    """使用 t-SNE 可视化 CNN 提取的特征（来自测试集）并保存"""
    print("\n--- 绘制和保存 t-SNE 特征分布图 ---")
    cnn_model.eval()
    ae_model.eval()

    all_features = []

    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            denoised_data = ae_model(data)

            # 前向传播，提取特征 (需要在 CNN 模型中记录 self.features)
            _ = cnn_model(denoised_data)
            all_features.append(cnn_model.features.cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    n_samples = min(500, features.shape[0])
    tsne_data = features[:n_samples]
    tsne_targets = np.array(all_targets)[:n_samples]
    tsne_predictions = np.array(all_predictions)[:n_samples]

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(tsne_data)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # 左图：真实标签
    scatter1 = ax[0].scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_targets, cmap='viridis', alpha=0.7)
    legend1 = ax[0].legend(handles=scatter1.legend_elements()[0], labels=CLASS_NAMES, title="True Class")
    ax[0].add_artist(legend1)
    ax[0].set_title(f't-SNE Visualization of CNN Features (True Labels, N={n_samples})')
    ax[0].grid(True)

    # 右图：预测标签
    scatter2 = ax[1].scatter(tsne_results[:, 0], tsne_results[:, 1], c=tsne_predictions, cmap='viridis', alpha=0.7)
    legend2 = ax[1].legend(handles=scatter2.legend_elements()[0], labels=CLASS_NAMES, title="Predicted Class")
    ax[1].add_artist(legend2)
    ax[1].set_title(f't-SNE Visualization of CNN Features (Predicted Labels, N={n_samples})')
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_features.png'))
    plt.close()


def visualize_feature_maps(cnn_model, ae_model, sample_img, device, output_dir):
    """可视化 CNN 早期卷积层和池化层的输出特征图并保存"""
    print("\n--- 绘制和保存 CNN 特征图 ---")
    cnn_model.eval()
    ae_model.eval()

    # ❗ 修复：确保输入张量具有批次维度 (Batch, Channel, Height, Width)
    # sample_img 是 (1, 64, 64)，需要 unsqueeze(0) 变成 (1, 1, 64, 64)
    if not torch.is_tensor(sample_img):
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        sample_tensor = transform(sample_img).unsqueeze(0).to(device)
    else:
        # 假设 sample_img 来自 test_loader[0] 是 (1, 64, 64)，需要添加批次维度
        sample_tensor = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        denoised_input = ae_model(sample_tensor)

        feature_maps = {}

        # Hook 函数: 捕获指定层的输出
        def hook_fn(module, input, output, name):
            # output 的形状是 (Batch, Channel, Height, Width)
            # 提取第一个样本的特征图 (Batch=0)，并移动到 CPU
            feature_maps[name] = output.squeeze(0).cpu()

        # 注册 Hooks
        hooks = [
            cnn_model.conv1.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv1')),
            cnn_model.pool.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'pool1')),
            cnn_model.conv2.register_forward_hook(lambda m, i, o: hook_fn(m, i, o, 'conv2')),
        ]

        # 执行前向传播
        _ = cnn_model(denoised_input)

        # 移除 Hooks
        for h in hooks:
            h.remove()

    # 绘制特征图
    for layer_name, feature_map_tensor in feature_maps.items():
        n_channels = feature_map_tensor.size(0)
        n_cols = 8
        n_rows = (n_channels + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 2 * n_rows))
        plt.suptitle(f"Feature Maps after {layer_name} ({n_channels} Channels)", fontsize=16)

        for i in range(n_channels):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            plt.imshow(feature_map_tensor[i].numpy(), cmap='jet')
            plt.title(f"C{i + 1}", fontsize=8)
            plt.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'feature_maps_{layer_name}.png'))
        plt.close()


# ======================================================================
#                            VI. 主执行模块
# ======================================================================

if __name__ == '__main__':
    # -------------------- A. 数据加载与初始化 --------------------
    train_loader, test_loader = get_dataloaders(DATA_PATH)

    autoencoder = Autoencoder().to(DEVICE)
    cnn_model = CNN().to(DEVICE)

    # 历史记录列表
    ae_losses = []
    train_losses = [];
    test_losses = []
    train_accuracies = [];
    test_accuracies = []
    test_precision = [];
    test_recall = [];
    test_f1 = []

    # -------------------- B. 自编码器训练与保存 --------------------
    print("\n" + "=" * 50);
    print("B. 阶段一：自编码器去噪训练");
    print("=" * 50)

    criterion_ae = nn.MSELoss()
    # AE 优化器不需要 L2 正则化
    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        loss = train_autoencoder_epoch(autoencoder, train_loader, criterion_ae, optimizer_ae, DEVICE, epoch)
        ae_losses.append(loss)

    torch.save(autoencoder.state_dict(), AE_MODEL_PATH)
    print(f"\n✅ 自编码器模型保存完毕: {AE_MODEL_PATH}")

    # -------------------- C. CNN 训练与测试 --------------------
    print("\n" + "=" * 50);
    print("C. 阶段二：CNN 分类训练与测试");
    print("=" * 50)

    criterion_cnn = nn.CrossEntropyLoss()
    # ✅ 改进点 2: 在 CNN 优化器中应用 WEIGHT_DECAY (L2 正则化)
    optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    try:
        # 加载 AE 模型权重以进行去噪
        autoencoder.load_state_dict(torch.load(AE_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"警告：无法加载 AE 模型权重，CNN 将无法去噪。错误: {e}")

    final_targets, final_predictions = None, None

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_cnn_epoch(cnn_model, autoencoder, train_loader, criterion_cnn, optimizer_cnn,
                                                DEVICE, epoch)
        train_losses.append(train_loss);
        train_accuracies.append(train_acc)

        test_loss, test_acc, p, r, f1, targets, preds = test_cnn_epoch(cnn_model, autoencoder, test_loader,
                                                                       criterion_cnn, DEVICE, epoch)
        test_losses.append(test_loss);
        test_accuracies.append(test_acc)
        test_precision.append(p);
        test_recall.append(r);
        test_f1.append(f1)

        final_targets, final_predictions = targets, preds

    torch.save(cnn_model.state_dict(), CNN_MODEL_PATH)
    print(f"\n✅ CNN 模型最终保存完毕: {CNN_MODEL_PATH}")

    # -------------------- D. 可视化分析与保存 --------------------
    print("\n" + "=" * 50);
    print("D. 阶段三：结果可视化与分析");
    print("=" * 50)

    # D.1 绘制去噪结果
    visualize_denoising(autoencoder, test_loader, DEVICE, OUTPUT_DIR, n_display=8)

    # D.2 绘制指标曲线
    history_data = [train_losses, test_losses, train_accuracies, test_accuracies, test_precision, test_recall, test_f1]
    plot_metrics(history_data, EPOCHS, OUTPUT_DIR)

    # D.3 t-SNE 特征图
    if final_targets is not None:
        plot_tsne_features(cnn_model, autoencoder, test_loader, DEVICE, final_targets, final_predictions, OUTPUT_DIR)

    # D.4 特征图可视化 (使用第一张测试图片)
    try:
        sample_batch, _ = next(iter(test_loader))
        # 传入 batch 中的第一个图片 (1, 64, 64)
        visualize_feature_maps(cnn_model, autoencoder, sample_batch[0], DEVICE, OUTPUT_DIR)
        print(f"\n✅ 所有可视化结果已成功保存到目录: {OUTPUT_DIR}")
    except Exception as e:
        print(f"\n❌ 特征图可视化失败: {e}. 请检查 DataLoader 是否为空或模型结构是否完整。")