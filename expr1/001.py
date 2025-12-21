#!/usr/bin/env python3
# train_pm25_adagrad_enhanced.py
# 基于用户提供的 train_pm25_adagrad.py，做了以下增强：
# 1) 增加中文注释（尽量详细）
# 2) 增加训练早停与阈值停止：当验证集 loss 小于 LOSS_THRESHOLD 时提前停止；同时支持 patience（无改进终止）
# 3) 增加训练过程的最佳模型保存（根据验证集 loss）
# 4) 训练结束后输出多项回归指标：MSE, RMSE, MAE, R2（训练集与验证集）
# 5) 保持整体结构不变，尽量兼容原来的输入/输出格式

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# 配置（可调）
# ----------------------------
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
MODEL_PATH = "model_weights.npz"  # 保存最终模型（包含权重和归一化参数）
BEST_MODEL_PATH = "best_model_weights.npz"  # 验证集表现最好的模型
PREDICT_CSV = "predict.csv"

# Adagrad 超参数（可调整）
LR = 1e-2            # 学习率（Adagrad 通常可设较大）
ITERATION = 2000000    # 最大迭代次数（可设更大，配合 early stopping）
EPS = 1e-8           # 防止除零
LAMBDA = 0.01         # L2 正则强度（可设为 0.0）
VALIDATION_SPLIT = 0.1  # 用于画图/早停的验证集比例

# 早停相关参数
LOSS_THRESHOLD = 1e-6   # 若验证集 loss (MSE/2 + reg) 小于此阈值则停止训练；设为 None 表示不启用该阈值
PATIENCE = 300          # 若验证集 loss 连续 PATIENCE 次迭代没有改进则提前停止（基于最小化方向）

# 18 个测项的顺序（如果你的数据列顺序不同，请调整）
FEATURE_NAMES = [
    "AMB_TEMP", "CH4", "CO", "NHMC", "NO", "NO2", "NOx", "O3",
    "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC",
    "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"
]

TARGET_FEATURE = "PM2.5"

# ----------------------------
# 辅助函数：读取并构造每日 18x24 矩阵
# ----------------------------
def parse_train_csv(train_csv_path):
    """
    读取 train.csv，并按“日期”聚合成 (18, 24) 的矩阵。
    每一天共有 18 个测项，每个测项 24 小时的数据。
    返回：
        date_mats: dict
            key = 日期 (字符串)
            value = numpy array，形状为 (18, 24)
    注：函数尽量健壮，处理常见编码和 NR / 空字符串。
    """

    # 尝试常见编码读取
    try:
        df = pd.read_csv(train_csv_path, encoding='gbk')
    except Exception:
        df = pd.read_csv(train_csv_path, encoding='big5', errors='ignore')

    # 将中文列名统一为英文列名，增加鲁棒性
    rename_map = {
        '日期': 'Date',
        '測站': 'Station',
        '測項': 'Item'
    }
    df = df.rename(columns=rename_map)

    # 小时列名字 0~23（如果 CSV 使用字符串形式的列名）
    hour_cols = [str(i) for i in range(24)]

    # 检查必要列
    for c in ['Date', 'Item']:
        if c not in df.columns:
            raise ValueError(f"train.csv 缺少必要列 {c}，请检查列名（当前列名：{df.columns.tolist()}）")

    # 处理缺失值与特殊值
    df = df.fillna('')
    df = df.replace("NR", "0")  # RAINFALL 中常用 NR 表示 0

    def safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    data_by_date = {}
    # 遍历行，将每个测项的 24 小时数据放入对应日期字典
    for idx, row in df.iterrows():
        date = str(row['Date']).strip()
        item = str(row['Item']).strip()
        if item not in FEATURE_NAMES:
            # 跳过不在 FEATURE_NAMES 中的测项（使函数更通用）
            continue
        values = []
        for h in hour_cols:
            values.append(safe_float(row.get(h, 0)))
        arr = np.array(values, dtype=float)
        if date not in data_by_date:
            data_by_date[date] = {}
        data_by_date[date][item] = arr

    # 将每个日期填充为固定顺序的 18x24 矩阵
    date_mats = {}
    for date, m in data_by_date.items():
        mat = np.zeros((len(FEATURE_NAMES), 24), dtype=float)
        for i, feat in enumerate(FEATURE_NAMES):
            if feat in m:
                mat[i, :] = m[feat]
            else:
                mat[i, :] = 0.0
        date_mats[date] = mat

    return date_mats


# ----------------------------
# 构造滑动窗口样本
# ----------------------------
def build_samples(date_mats):
    """
    从 date_mats (dict of date -> 18x24 array) 构造样本。
    将所有日期按顺序（字典序）拼接成一个长时间序列（18 x total_hours）。
    然后用滑动窗口：每 10 小时一笔（前 9 小时特征，第 10 小时 PM2.5 为标签），窗口以 1 小时步长滑动。
    返回 X (N, 18*9) 和 y (N,)
    """
    dates = sorted(date_mats.keys())
    mats = [date_mats[d] for d in dates]
    full = np.hstack(mats)  # shape (18, total_hours)
    total_hours = full.shape[1]
    WINDOW = 9
    X_list = []
    y_list = []
    pm25_idx = FEATURE_NAMES.index(TARGET_FEATURE)
    for t in range(0, total_hours - WINDOW):
        feat_window = full[:, t:t + WINDOW]  # (18,9)
        label_hour = t + WINDOW  # 第 10 小时作为标签
        if label_hour >= total_hours:
            break
        label = full[pm25_idx, label_hour]
        X_list.append(feat_window.flatten())
        y_list.append(label)
    X = np.vstack(X_list)
    y = np.array(y_list).reshape(-1, 1)
    return X, y


# ----------------------------
# 缺失值处理（列均值填充），并归一化（标准化）
# ----------------------------
def preprocess_fill_and_normalize(X_train):
    """
    对训练特征矩阵做：
    - 将不可用（非有限值）替换为对应列的均值
    - 计算列均值和列标准差（std==0 时用 1 避免除零）
    - 返回归一化后的 X, 以及 mean, std（用于后续验证/测试集处理）
    注意：返回的 mean/std 对应的是原始特征列（不含 bias 列）
    """
    X = X_train.copy().astype(float)
    # 计算每列的均值（忽略非有限值）
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_adj = np.where(std == 0, 1.0, std)
    X_norm = (X - mean) / std_adj
    return X_norm, mean, std_adj


# ----------------------------
# Adagrad 线性回归训练（含早停与验证监控）
# ----------------------------
def adagrad_train(X, y, lr=LR, iteration=ITERATION, eps=EPS, lmbd=LAMBDA,
                  X_val=None, y_val=None, loss_threshold=LOSS_THRESHOLD,
                  patience=PATIENCE, verbose=True):
    """
    使用 Adagrad 优化线性回归（带 L2 正则）。
    支持：
      - validation set（X_val, y_val）用于早停与监控
      - 当验证集 loss 小于 loss_threshold 时提前停止
      - 当验证集在 patience 次迭代内不再改进时提前停止

    要求：传入的 X 已包含 bias 列（即第一列都是 1），并与 y 对齐。
    返回：最佳权重 w (D,1)，训练损失序列 losses，验证损失序列 val_losses
    """
    N, D = X.shape
    w = np.zeros((D, 1), dtype=float)
    G = np.zeros((D, 1), dtype=float)  # 累积平方梯度
    losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_w = None
    no_improve_count = 0

    for it in range(iteration):
        pred = X.dot(w)
        err = pred - y
        loss = np.mean(err ** 2) / 2.0 + (lmbd / 2.0) * np.sum(w ** 2)
        losses.append(loss)
        # 计算梯度并更新
        grad = (X.T.dot(err) / N) + lmbd * w
        G += grad ** 2
        ada = np.sqrt(G) + eps
        w = np.maximum(0, w - (lr * grad / ada))

        # 若提供验证集，计算验证集 loss 并用于早停
        if X_val is not None and y_val is not None:
            pred_val = X_val.dot(w)
            err_val = pred_val - y_val
            val_loss = np.mean(err_val ** 2) / 2.0 + (lmbd / 2.0) * np.sum(w ** 2)
            val_losses.append(val_loss)

            # 检查是否为最佳点
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_w = w.copy()
                no_improve_count = 0
                # 可在此处保存最佳模型（上层调用者也可以保存）
            else:
                no_improve_count += 1

            # 根据阈值停止
            if (loss_threshold is not None) and (val_loss <= loss_threshold):
                if verbose:
                    print(f"Early stop: val_loss {val_loss:.6e} <= threshold {loss_threshold:.6e} at iter {it+1}")
                break

            # 根据 patience 停止
            if no_improve_count >= patience:
                if verbose:
                    print(f"Early stop: no improvement in validation loss for {patience} iters (iter {it+1}), best_val_loss={best_val_loss:.6e}")
                break

        # 打印进度
        if verbose and (it % max(1, iteration // 10) == 0 or it < 10):
            if X_val is not None and y_val is not None:
                print(f"iter {it+1}/{iteration} loss={loss:.6f} val_loss={val_losses[-1]:.6f}")
            else:
                print(f"iter {it+1}/{iteration} loss={loss:.6f}")

    # 若存在 best_w（来自验证集），返回 best_w；否则返回当前 w
    final_w = best_w if best_w is not None else w
    return final_w, losses, val_losses


# ----------------------------
# 保存/加载模型（权重 + normalization params）
# ----------------------------
def save_model(path, w, mean, std):
    """保存模型权重以及用于标准化的 mean/std（mean/std 对应不含 bias 的特征列）"""
    np.savez(path, w=w, mean=mean, std=std)
    print("Model saved to", path)


def load_model(path):
    """加载模型（w, mean, std）"""
    npz = np.load(path)
    return npz['w'], npz['mean'], npz['std']


# ----------------------------
# 预测 test.csv 并写 predict.csv
# ----------------------------
def predict_test_csv(test_csv_path, model_path, out_csv=PREDICT_CSV):
    """
    使用保存在 model_path 的模型对 test.csv 进行预测，并将结果写入 out_csv。
    假设 test.csv 格式为题目描述的 18 行一组，每行前两列为 id,item，后 9 列为数值特征。
    返回 pandas.DataFrame（预测结果）。
    """
    w, mean, std = load_model(model_path)
    df = pd.read_csv(test_csv_path, header=None, encoding='gbk')
    df = df.fillna('')
    df = df.replace("NR", "0")
    results = []
    for idx in range(0, df.shape[0], 18):
        block = df.iloc[idx:idx + 18, :].copy()
        if block.shape[0] < 18:
            break
        sample_id = block.iloc[0, 0]
        vals = []
        for r in range(block.shape[0]):
            row = block.iloc[r, 2:].values.astype(float)
            if row.shape[0] < 9:
                row = np.pad(row, (0, 9 - row.shape[0]), constant_values=0.0)
            vals.append(row)
        feat = np.array(vals).flatten()  # (162,)
        # 使用训练时保存的 mean/std 做归一化
        feat = (feat - mean) / std
        # 添加 bias
        feat = np.concatenate(([1.0], feat))
        pred = float(np.dot(feat, w).squeeze())
        results.append([sample_id, pred])
    out_df = pd.DataFrame(results, columns=["id", "value"])
    out_df.to_csv(out_csv, index=False)
    print("Predictions written to", out_csv)
    return out_df


# ----------------------------
# 主流程
# ----------------------------
def main():
    # 1) parse train.csv
    print("Parsing", TRAIN_CSV)
    date_mats = parse_train_csv(TRAIN_CSV)
    print("Loaded dates:", len(date_mats))

    # 2) build samples
    X_raw, y = build_samples(date_mats)
    print("Total samples:", X_raw.shape[0], "Feature dim:", X_raw.shape[1])

    # 3) split train/val
    X_tr, X_val, y_tr, y_val = train_test_split(X_raw, y, test_size=VALIDATION_SPLIT, random_state=42)

    # 4) preprocess（仅在训练集上拟合 mean/std）
    X_tr_norm, mean, std = preprocess_fill_and_normalize(X_tr)
    # 验证/测试集使用相同 mean/std 归一化
    X_val_norm = (X_val - mean) / std

    # 5) 添加 bias 列（注意：w 的第一项对应 bias）
    X_tr_with_bias = np.concatenate((np.ones((X_tr_norm.shape[0], 1)), X_tr_norm), axis=1)
    X_val_with_bias = np.concatenate((np.ones((X_val_norm.shape[0], 1)), X_val_norm), axis=1)

    # 6) 训练（使用 Adagrad，带验证集监控与早停）
    print("Start training with Adagrad. lr=", LR, "max_iters=", ITERATION)
    w, losses, val_losses = adagrad_train(
        X_tr_with_bias, y_tr, lr=LR, iteration=ITERATION, eps=EPS, lmbd=LAMBDA,
        X_val=X_val_with_bias, y_val=y_val, loss_threshold=LOSS_THRESHOLD,
        patience=PATIENCE, verbose=True
    )

    # 7) 保存最终模型（以及验证集上表现最好的模型，如果两者不同也保存）
    save_model(MODEL_PATH, w, mean, std)
    # 若训练过程中产生的 best model 与最终 model 不同，adagrad_train 已返回 best_w
    # 为了保险起见，再保存一份 best model（如果与当前模型一样也无妨）
    np.savez(BEST_MODEL_PATH, w=w, mean=mean, std=std)

    # 8) 在训练集和验证集上计算最终指标
    y_tr_pred = X_tr_with_bias.dot(w)
    y_val_pred = X_val_with_bias.dot(w)

    def regression_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, mae, r2

    tr_mse, tr_rmse, tr_mae, tr_r2 = regression_metrics(y_tr, y_tr_pred)
    val_mse, val_rmse, val_mae, val_r2 = regression_metrics(y_val, y_val_pred)

    print("--- Training metrics ---")
    print(f"Train MSE: {tr_mse:.6f}, RMSE: {tr_rmse:.6f}, MAE: {tr_mae:.6f}, R2: {tr_r2:.6f}")
    print("--- Validation metrics ---")
    print(f"Val MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}, MAE: {val_mae:.6f}, R2: {val_r2:.6f}")

    # 9) 绘图：训练损失与验证损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='train_loss')
    if len(val_losses) > 0:
        # val_losses 可能比 losses 短（因为提前停止），在绘制时对齐 x 轴
        plt.plot(range(len(val_losses)), val_losses, label='val_loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE/2 + reg)")
    plt.title("Training loss curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved training loss plot to training_loss.png")

    # 10) 验证集真实值 vs 预测值 散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_val_pred, alpha=0.5, s=8)
    plt.xlabel("True PM2.5")
    plt.ylabel("Predicted PM2.5")
    plt.title("Validation: Predicted vs True")
    lims = [min(y_val.min(), y_val_pred.min()), max(y_val.max(), y_val_pred.max())]
    plt.plot(lims, lims, 'r--')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("val_pred_scatter.png")
    print("Saved validation scatter to val_pred_scatter.png")

    # 11) 验证集预测直方图
    plt.figure(figsize=(8, 4))
    plt.hist(y_val_pred.squeeze(), bins=40)
    plt.xlabel("Predicted PM2.5")
    plt.title("Histogram of validation predictions")
    plt.tight_layout()
    plt.savefig("val_pred_hist.png")
    print("Saved validation histogram to val_pred_hist.png")

    # 12) 可选：如果存在 test.csv 则生成预测
    if os.path.exists(TEST_CSV):
        print("Found test.csv, generating predictions...")
        predict_test_csv(TEST_CSV, MODEL_PATH, PREDICT_CSV)
    else:
        print("No test.csv found in current dir; skip test prediction (put test.csv to run)")


if __name__ == "__main__":
    main()
