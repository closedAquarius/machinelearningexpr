#!/usr/bin/env python3
# train_pm25_adagrad.py
# 适用于题目描述的数据格式：train.csv 每天为 18 行（每行 0-23 小时），test.csv 为按 9 小时特征的样本（id, item, 9 values）

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ----------------------------
# 配置（可调）
# ----------------------------
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
MODEL_PATH = "model_weights.npz"
PREDICT_CSV = "predict.csv"

# Adagrad hyperparams (可调整)
LR = 1e-1  # 学习率（Adagrad 通常可用较大 lr）
ITERATION = 5000  # 迭代次数
EPS = 1e-8  # 防止除零
LAMBDA = 0.0  # L2 正则强度（可设为 0.0）
VALIDATION_SPLIT = 0.05  # 用于画图的验证比例

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
    """

    # 尝试用常见的编码读取
    try:
        df = pd.read_csv(train_csv_path, encoding='gbk')
    except:
        df = pd.read_csv(train_csv_path, encoding='big5', errors='ignore')

    # ================================
    # 1. 将中文列名统一为英文列名
    # ================================
    rename_map = {
        '日期': 'Date',
        '測站': 'Station',
        '測項': 'Item'
    }
    df = df.rename(columns=rename_map)

    # ================================
    # 2. 基本检查
    # ================================
    hour_cols = [str(i) for i in range(24)]  # 小时列名 0~23

    # 检查必要列
    for c in ['Date', 'Item']:
        if c not in df.columns:
            raise ValueError(f"train.csv 缺少必要列 {c}，请检查列名（当前列名：{df.columns.tolist()}）")

    # ================================
    # 3. 处理缺失值
    # ================================
    df = df.fillna('')        # 空缺值填为空字符串
    df = df.replace("NR", "0")   # 将 RAINFALL 中常见的 NR 替换为 0

    # 安全的 float 转换
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0

    # ================================
    # 4. 按日期聚合，构造 18 x 24 的矩阵
    # ================================
    data_by_date = {}

    # FEATURE_NAMES 必须在外部定义
    # 例如：
    # FEATURE_NAMES = ["AMB_TEMP", "CH4", "CO", ... 共 18 项]
    global FEATURE_NAMES

    for idx, row in df.iterrows():
        date = str(row['Date']).strip()
        item = str(row['Item']).strip()

        # 如果 CSV 中的测项不在你的 FEATURE_NAMES 中，则跳过
        if item not in FEATURE_NAMES:
            continue

        # 提取 24 小时的数据
        values = []
        for h in hour_cols:
            values.append(safe_float(row.get(h, 0)))  # 若缺失小时列，用 0

        arr = np.array(values, dtype=float)  # shape (24,)

        # 保存
        if date not in data_by_date:
            data_by_date[date] = {}
        data_by_date[date][item] = arr

    # ================================
    # 5. 转换为每天固定的 18 × 24 矩阵
    # ================================
    date_mats = {}

    for date, m in data_by_date.items():
        mat = np.zeros((len(FEATURE_NAMES), 24), dtype=float)

        for i, feat in enumerate(FEATURE_NAMES):
            if feat in m:
                mat[i, :] = m[feat]
            else:
                # 若缺某个测项，可填 0 或后续再做均值填补
                mat[i, :] = 0.0

        date_mats[date] = mat

    return date_mats



# ----------------------------
# 构造滑动窗口样本
# ----------------------------
def build_samples(date_mats):
    """
    从 date_mats (dict of date -> 18x24 array) 构造样本。
    将所有日期按顺序（字典顺序）拼接成一个长时间序列（18 x total_hours）。
    然后用滑动窗口：每 10 小时一笔（前 9 小时特征， 第 10 小时 PM2.5 为标签），窗口以 1 小时步长滑动。
    返回 X (N, 18*9) 和 y (N,)
    """
    # Sort dates to preserve chronological order (假定 date 字符串能按字典序排序）
    dates = sorted(date_mats.keys())
    # Concatenate horizontally across dates
    mats = [date_mats[d] for d in dates]
    # mats is list of arrays shape (18,24)
    full = np.hstack(mats)  # shape (18, total_hours)
    total_hours = full.shape[1]
    WINDOW = 9
    STEP = 1
    X_list = []
    y_list = []
    pm25_idx = FEATURE_NAMES.index(TARGET_FEATURE)
    # we need label at time t+9 (0-based), features are t..t+8
    for t in range(0, total_hours - WINDOW):
        feat_window = full[:, t:t + WINDOW]  # shape (18,9)
        label_hour = t + WINDOW  # index of the 10th hour
        # ensure label is available
        if label_hour >= total_hours:
            break
        # label is PM2.5 at label_hour
        label = full[pm25_idx, label_hour]
        X_list.append(feat_window.flatten())  # (18*9,)
        y_list.append(label)
    X = np.vstack(X_list)  # (N, 162)
    y = np.array(y_list).reshape(-1, 1)  # (N,1)
    return X, y


# ----------------------------
# 缺失值处理（列均值填充），并归一化（标准化）
# ----------------------------
def preprocess_fill_and_normalize(X_train):
    """
    替换缺失（NaN）并对每一列进行标准化（减均值除以标准差）。
    返回 X_norm, mean, std
    """
    X = X_train.copy().astype(float)
    # NaN -> column mean
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    # replace nan with col_mean; if col mean nan (all nan), set 0
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_adj = np.where(std == 0, 1.0, std)
    X_norm = (X - mean) / std_adj
    return X_norm, mean, std_adj


# ----------------------------
# Adagrad 线性回归训练
# ----------------------------
def adagrad_train(X, y, lr=LR, iteration=ITERATION, eps=EPS, lmbd=LAMBDA, verbose=True):
    """
    X: (N, D) - features (assumed already normalized and with bias column if desired)
    y: (N, 1)
    returns w: (D,1), losses list
    """
    N, D = X.shape
    w = np.zeros((D, 1), dtype=float)
    G = np.zeros((D, 1), dtype=float)  # 累积平方梯度
    losses = []
    for it in range(iteration):
        pred = X.dot(w)  # (N,1)
        err = pred - y
        loss = np.mean(err ** 2) / 2.0 + (lmbd / 2.0) * np.sum(w ** 2)
        losses.append(loss)
        # gradient (D,1)
        grad = (X.T.dot(err) / N) + lmbd * w
        G += grad ** 2
        # Adagrad update
        ada = np.sqrt(G) + eps
        w = w - (lr * grad / ada)
        if verbose and (it % max(1, iteration // 10) == 0 or it < 10):
            print(f"iter {it + 1}/{iteration} loss={loss:.6f}")
    return w, losses


# ----------------------------
# 保存/加载模型（权重 + normalization params）
# ----------------------------
def save_model(path, w, mean, std):
    np.savez(path, w=w, mean=mean, std=std)
    print("Model saved to", path)


def load_model(path):
    npz = np.load(path)
    return npz['w'], npz['mean'], npz['std']


# ----------------------------
# 预测 test.csv 并写 predict.csv
# ----------------------------
def predict_test_csv(test_csv_path, model_path, out_csv=PREDICT_CSV):
    # load model
    w, mean, std = load_model(model_path)
    df = pd.read_csv(test_csv_path, header=None, encoding='gbk')
    # ===== 加上与训练集一致的缺失值处理 =====
    df = df.fillna('')
    df = df.replace("NR", "0")
    # test.csv 按题意：每 18 行为一组，第一列 id，第二列 item，后 9 列为特征
    # 为了兼容多种格式，这里假定：每行是 id, item, v0..v8  (共 11 列)
    results = []
    for idx in range(0, df.shape[0], 18):
        block = df.iloc[idx:idx + 18, :].copy()
        if block.shape[0] < 18:
            break
        # id 列取第一行第一列
        sample_id = block.iloc[0, 0]
        # read the 9 cols starting from column index 2
        vals = []
        for r in range(block.shape[0]):
            row = block.iloc[r, 2:].values.astype(float)
            # if fewer than 9 values, pad zeros
            if row.shape[0] < 9:
                row = np.pad(row, (0, 9 - row.shape[0]), constant_values=0.0)
            vals.append(row)
        feat = np.array(vals).flatten()  # (162,)
        # preprocess using saved mean/std
        feat = (feat - mean) / std
        # add bias (if model expects bias in first column)
        feat = np.concatenate(([1.0], feat))  # assumes model was trained with bias as first column
        pred = float(np.dot(feat, w).squeeze())
        results.append([sample_id, pred])
    # write csv
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
    # 2) build samples X,y
    X_raw, y = build_samples(date_mats)
    print("Total samples:", X_raw.shape[0], "Feature dim:", X_raw.shape[1])
    # 3) split train/val for monitoring
    X_tr, X_val, y_tr, y_val = train_test_split(X_raw, y, test_size=VALIDATION_SPLIT, random_state=42)
    # 4) preprocess (fit on train)
    X_tr_norm, mean, std = preprocess_fill_and_normalize(X_tr)
    # add bias column as first column of ones
    X_tr_with_bias = np.concatenate((np.ones((X_tr_norm.shape[0], 1)), X_tr_norm), axis=1)
    # 5) train Adagrad
    print("Start training with Adagrad. lr=", LR, "iters=", ITERATION)
    w, losses = adagrad_train(X_tr_with_bias, y_tr, lr=LR, iteration=ITERATION, eps=EPS, lmbd=LAMBDA, verbose=True)
    # 6) save model (store weights and normalization params)
    save_model(MODEL_PATH, w, mean, std)
    # 7) validation prediction for visualization
    X_val_norm = (X_val - mean) / std
    X_val_with_bias = np.concatenate((np.ones((X_val_norm.shape[0], 1)), X_val_norm), axis=1)
    y_val_pred = X_val_with_bias.dot(w)
    # compute val loss (MSE)
    val_mse = np.mean((y_val_pred - y_val) ** 2)
    print("Validation MSE:", val_mse)
    # 8) plot training loss
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE/2 + reg)")
    plt.title("Training loss curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved training loss plot to training_loss.png")
    # 9) plot predicted vs true on validation
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
    # 10) histogram of predictions
    plt.figure(figsize=(8, 4))
    plt.hist(y_val_pred.squeeze(), bins=40)
    plt.xlabel("Predicted PM2.5")
    plt.title("Histogram of validation predictions")
    plt.tight_layout()
    plt.savefig("val_pred_hist.png")
    print("Saved validation histogram to val_pred_hist.png")
    # 11) optional: run test predict if test.csv present
    if os.path.exists(TEST_CSV):
        print("Found test.csv, generating predictions...")
        predict_test_csv(TEST_CSV, MODEL_PATH, PREDICT_CSV)
    else:
        print("No test.csv found in current dir; skip test prediction (put test.csv to run)")


if __name__ == "__main__":
    main()
