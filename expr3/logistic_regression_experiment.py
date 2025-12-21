import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split  # 🌟 引入分割工具
import os

# --- 1. 可调参数集中 (要求5 + 优化参数) ---
MODEL_PARAMS = {
    'learning_rate': 0.1,  # 学习率
    'num_iterations': 15000,  # 最大迭代次数
    'reg_lambda': 0.01,  # L2 正则化参数

    # 🌟 早停参数 (基于验证集 Loss 连续不改进)
    'early_stopping_patience': 500,  # 连续多少轮 Validation Loss 无改进时停止
    'loss_threshold': 0.35,  # Training Loss 小于此值时停止 (保留作为加速训练的硬性阈值)
    'min_delta': 1e-5,  # 判定为"改进"的最小 Loss 变化量
    'print_interval': 100,  # 打印 Loss 的间隔轮数

    # 🌟 验证集参数
    'validation_split_ratio': 0.2  # 划分 20% 原始训练数据作为验证集
}

# 文件路径和设置
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
OUTPUT_DIR = 'output'
PREDICT_FILE = os.path.join(OUTPUT_DIR, 'predict.csv')


# --- 2. 数据预处理模块 (不变) ---

def load_data(train_file, test_file):
    """
    加载训练和测试数据集。
    处理数据中可能存在的问号 '?' 或空格 ' ?'，将其替换为 NaN。
    """
    print("-> 正在加载数据...")
    train_df = pd.read_csv(train_file, na_values=[' ?', '?'])
    test_df = pd.read_csv(test_file, na_values=[' ?', '?'])

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    train_df['income'] = train_df['income'].str.strip()

    return train_df, test_df


def preprocess_data(train_df, test_df):
    """
    对训练集和测试集进行数据预处理。
    """
    Y_train = (train_df['income'] == '>50K').astype(int).values
    train_df = train_df.drop('income', axis=1)

    continuous_features = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    discrete_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                         'native_country']

    for col in discrete_features:
        mode_val = train_df[col].mode()[0]
        train_df[col] = train_df[col].fillna(mode_val)
        test_df[col] = test_df[col].fillna(mode_val)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = pd.get_dummies(combined_df, columns=discrete_features)

    X_train_processed = combined_df.iloc[:len(train_df)]
    X_test_processed = combined_df.iloc[len(train_df):]

    scaler = MinMaxScaler()

    X_train_processed = X_train_processed.copy()
    X_test_processed = X_test_processed.copy()

    X_train_processed.loc[:, continuous_features] = scaler.fit_transform(X_train_processed[continuous_features])
    X_test_processed.loc[:, continuous_features] = scaler.transform(X_test_processed[continuous_features])

    # --- 转换为 numpy 数组并添加偏置项 (Bias Term) ---
    X_train_full = np.insert(X_train_processed.values, 0, 1, axis=1)  # 原始训练集（包含验证集）
    X_test = np.insert(X_test_processed.values, 0, 1, axis=1)

    feature_names = ['Bias'] + list(X_train_processed.columns)

    return X_train_full, Y_train, X_test, feature_names


# --- 3. Logistic 回归核心函数 (不变) ---

def sigmoid(z):
    """Sigmoid 激活函数"""
    z = np.array(z, dtype=np.float64)
    return 1 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def compute_loss(X, Y, W, reg_lambda=0.0):
    """计算交叉熵损失"""
    m = X.shape[0]
    h = sigmoid(X @ W)
    loss = (-1 / m) * np.sum(Y * np.log(h + 1e-10) + (1 - Y) * np.log(1 - h + 1e-10))
    regularization_term = (reg_lambda / (2 * m)) * np.sum(W[1:] ** 2)
    return loss + regularization_term


def compute_gradient(X, Y, W, reg_lambda=0.0):
    """计算带 L2 正则化的梯度"""
    m = X.shape[0]
    h = sigmoid(X @ W)
    gradient = (1 / m) * (X.T @ (h - Y))
    regularization_gradient = np.zeros_like(W)
    regularization_gradient[1:] = (reg_lambda / m) * W[1:]
    return gradient + regularization_gradient


def gradient_descent(X_train, Y_train, X_val, Y_val, params):
    """
    🌟 修改: 使用验证集 X_val, Y_val 进行早停和 Loss 记录。
    """
    # 训练参数
    learning_rate = params['learning_rate']
    num_iterations = params['num_iterations']
    reg_lambda = params['reg_lambda']
    print_interval = params['print_interval']

    # 早停参数
    patience = params['early_stopping_patience']
    loss_threshold = params['loss_threshold']
    min_delta = params['min_delta']

    # 初始化权重 W 和 Loss 历史记录
    W = np.zeros(X_train.shape[1], dtype=np.float64)
    train_loss_history = []
    val_loss_history = []

    best_val_loss = np.inf
    wait = 0
    best_W = W.copy()  # 保存验证集上表现最好的权重

    print(f"  Learning Rate: {learning_rate}, L2 Lambda: {reg_lambda}, Max Iters: {num_iterations}")
    print(f"  Early Stopping (Val Loss): Patience={patience}, Min Delta={min_delta}")
    print(f"  Training Set Size: {X_train.shape[0]}, Validation Set Size: {X_val.shape[0]}")

    for i in range(num_iterations):
        # 1. 计算梯度和更新权重
        gradient = compute_gradient(X_train, Y_train, W, reg_lambda)
        W = W - learning_rate * gradient

        # 2. 计算当前 Loss 并记录
        train_loss = compute_loss(X_train, Y_train, W, reg_lambda)
        val_loss = compute_loss(X_val, Y_val, W, reg_lambda)  # 🌟 计算验证集 Loss

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # 3. 打印 Loss
        if (i + 1) % print_interval == 0:
            print(f"  Iteration {i + 1}/{num_iterations}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # 4. 早停检查 (基于验证集 Loss)
        # 优先使用训练 Loss 阈值快速停止（如果 Loss 够低）
        if train_loss < loss_threshold:
            print(
                f"\n  *** 早停触发 (Train Loss): Loss ({train_loss:.6f}) 小于阈值 ({loss_threshold:.6f})。在第 {i + 1} 轮停止。***")
            best_W = W.copy()
            break

        # 基于验证集 Loss 改进
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_W = W.copy()  # 🌟 更新最佳权重
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"\n  *** 早停触发 (Val Loss): 验证集 Loss 连续 {patience} 轮无改进 ({best_val_loss:.6f})。在第 {i + 1} 轮停止。***")
                W = best_W  # 🌟 恢复到最佳权重
                break

    # 如果循环正常结束，说明达到了最大迭代次数
    else:
        print(f"\n  *** 训练结束: 达到最大迭代次数 {num_iterations}。***")
        W = best_W  # 如果达到最大迭代，仍使用历史最好的权重

    return W, train_loss_history, val_loss_history  # 🌟 返回两条 Loss 历史


def predict(X, W):
    """预测类别标签 (0 或 1)"""
    probabilities = sigmoid(X @ W)
    predictions = (probabilities >= 0.4).astype(int)
    return predictions


def calculate_metrics(Y_true, Y_pred, X, W):
    """计算模型性能指标：准确率 (Accuracy)、F1-Score、RMSE"""
    accuracy = np.mean(Y_true == Y_pred)

    TP = np.sum((Y_true == 1) & (Y_pred == 1))
    FP = np.sum((Y_true == 0) & (Y_pred == 1))
    FN = np.sum((Y_true == 1) & (Y_pred == 0))

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))

    return accuracy, f1_score, rmse, precision, recall


# --- 4. 可视化和分析函数 (要求6, 7) ---

def plot_loss(train_loss_history, val_loss_history):
    """🌟 修改: 可视化训练集和验证集的损失函数曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_history, label='Training Loss', color='blue')
    plt.plot(val_loss_history, label='Validation Loss', color='orange')

    # 标注早停点（如果发生）
    if len(train_loss_history) < len(val_loss_history):
        # 这种情况通常不发生，除非训练 loss 阈值更早停止
        stop_point = len(train_loss_history)
    else:
        # 早停点发生在最后一次迭代
        stop_point = len(train_loss_history)

    plt.axvline(x=stop_point - 1, color='r', linestyle='--', label='Stopping Point')

    plt.title('Loss Function Curve (Training vs. Validation)')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve_with_val.png'))  # 🌟 文件名修改
    plt.close()
    print("-> Loss 曲线图 (包含验证集 Loss) 已保存到 output/loss_curve_with_val.png")


# ... (plot_weight_heatmap, plot_tsne, analyze_feature_contribution 保持不变)

def plot_weight_heatmap(W, feature_names):
    """可视化学到的参数（权重 W）的条形图"""
    W_abs = np.abs(W[1:])
    feature_names_no_bias = np.array(feature_names[1:])
    N = 30
    ind = np.argsort(W_abs)[::-1][:N]
    top_W = W[1:][ind]
    top_features = feature_names_no_bias[ind]
    plt.figure(figsize=(12, 8))
    plt.barh(top_features, top_W, color=['g' if w > 0 else 'r' for w in top_W])
    plt.title(f'Top {N} Feature Weights (Contribution to Log-Odds)')
    plt.xlabel('Weight Value')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_weights.png'))
    plt.close()
    print("-> 权重贡献条形图已保存到 output/feature_weights.png")


def plot_tsne(X_train, Y_train):
    """使用 t-SNE 降维可视化训练集数据分布和类别分离情况"""
    print("-> 正在进行 t-SNE 降维 (可能需要一些时间)...")
    N_sample = 5000
    indices = np.random.choice(X_train.shape[0], N_sample, replace=False)
    X_sample = X_train[indices]
    Y_sample = Y_train[indices]
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    X_tsne = tsne.fit_transform(X_sample)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[Y_sample == 0, 0], X_tsne[Y_sample == 0, 1],
                label='<=50K (0)', alpha=0.6, s=10, color='blue')
    plt.scatter(X_tsne[Y_sample == 1, 0], X_tsne[Y_sample == 1, 1],
                label='>50K (1)', alpha=0.6, s=10, color='red')
    plt.title(f't-SNE Visualization of Training Data (Sampled N={N_sample})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_visualization.png'))
    plt.close()
    print("-> t-SNE 降维图已保存到 output/tsne_visualization.png")


def analyze_feature_contribution(W, feature_names):
    """分析并打印特征贡献度（要求7）"""
    print("\n--- 7. 特征贡献度分析 (基于权重 W 的绝对值) ---")
    ind = np.argsort(np.abs(W))[::-1]
    print(f"| {'Feature Name'.ljust(40)} | {'Weight (W)'.ljust(15)} |")
    print("-" * 60)
    for i in ind[:20]:
        feature = feature_names[i]
        weight = W[i]
        print(f"| {feature.ljust(40)} | {weight:15.6f} |")


# --- 5. 主执行函数 ---

def main():
    """
    主程序流程
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("--- 1. 数据加载与预处理 ---")
    train_df, test_df = load_data(TRAIN_FILE, TEST_FILE)
    X_train_full, Y_train_full, X_test, feature_names = preprocess_data(train_df, test_df)

    # 🌟 关键修改：将训练集进一步分割为训练集和验证集 (80%/20%)
    split_ratio = MODEL_PARAMS['validation_split_ratio']
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_full, Y_train_full, test_size=split_ratio, random_state=42, stratify=Y_train_full
    )

    print(f"数据预处理完成。特征维度 (包括偏置项): {X_train_full.shape[1]}")
    print(f"分割后的训练集形状 X_train: {X_train.shape}")
    print(f"分割后的验证集形状 X_val: {X_val.shape}")

    print("\n--- 2. 模型训练 (梯度下降) ---")
    # 🌟 传入验证集
    W, train_loss_history, val_loss_history = gradient_descent(X_train, Y_train, X_val, Y_val, MODEL_PARAMS)

    print(f"\n训练完成。最终训练集损失: {train_loss_history[-1]:.4f}")
    print(f"训练完成。最终验证集损失: {val_loss_history[-1]:.4f}")

    print("\n--- 3. 性能评估 (要求6) ---")
    # 在验证集上评估模型性能
    Y_pred_val = predict(X_val, W)
    accuracy, f1_score, rmse, precision, recall = calculate_metrics(Y_val, Y_pred_val, X_val, W)

    print(f"在验证集上的性能指标 (用于模型选择):")
    print(f"  - 准确率 (Accuracy): {accuracy:.4f}")
    print(f"  - 精确率 (Precision): {precision:.4f}")
    print(f"  - 召回率 (Recall): {recall:.4f}")
    print(f"  - F1-Score: {f1_score:.4f}")
    print(f"  - 均方根误差 (RMSE): {rmse:.4f}")

    print("\n--- 4. 模型预测与结果输出 (要求4) ---")
    Y_pred_test = predict(X_test, W)

    ids = np.arange(1, len(Y_pred_test) + 1)
    results_df = pd.DataFrame({
        'id': ids,
        'label': Y_pred_test
    })
    results_df.to_csv(PREDICT_FILE, index=False)
    print(f"-> 预测结果已保存到 {PREDICT_FILE}")

    # --- 5. 可视化 (要求6) ---
    plot_loss(train_loss_history, val_loss_history)  # 🌟 传入两个 Loss 历史
    plot_weight_heatmap(W, feature_names)
    plot_tsne(X_train_full, Y_train_full)  # t-SNE 仍使用全量训练集（包含验证集）进行可视化

    # --- 6. 特征贡献分析 (要求7) ---
    analyze_feature_contribution(W, feature_names)

    print("\n--- 实验结束 ---")


if __name__ == '__main__':
    main()