#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_pm25_adagrad_fixed.py
# 修复 NaN/inf 问题，滑动窗口 9 小时预测第 10 小时 PM2.5，训练在 log1p 空间，评估在原空间

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------------
# 配置
# ----------------------------
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
MODEL_PATH = "model_weights.npz"
BEST_MODEL_PATH = "best_model_weights.npz"
PREDICT_CSV = "predict.csv"

LR = 0.1
ITERATION = 200000
EPS = 1e-8
LAMBDA = 0.0
VALIDATION_SPLIT = 0.05
PATIENCE = 200
LOSS_THRESHOLD = None

FEATURE_NAMES = [
    "AMB_TEMP", "CH4", "CO", "NHMC", "NO", "NO2", "NOx", "O3",
    "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC",
    "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR"
]
TARGET_FEATURE = "PM2.5"

# ----------------------------
# 解析 train.csv
# ----------------------------
def parse_train_csv(train_csv_path):
    try:
        df = pd.read_csv(train_csv_path, encoding='gbk')
    except:
        df = pd.read_csv(train_csv_path, encoding='big5', errors='ignore')
    rename_map = {'日期': 'Date', '測站': 'Station', '測項': 'Item'}
    df = df.rename(columns=rename_map)
    hour_cols = [str(i) for i in range(24)]
    for c in ['Date', 'Item']:
        if c not in df.columns:
            raise ValueError(f"train.csv 缺少 {c} 列")
    df = df.fillna(0).replace("NR", 0)
    def safe_float(x):
        try:
            return float(x)
        except:
            return 0.0
    data_by_date = {}
    for idx, row in df.iterrows():
        date = str(row['Date']).strip()
        item = str(row['Item']).strip()
        if item not in FEATURE_NAMES:
            continue
        values = [safe_float(row.get(h, 0.0)) for h in hour_cols]
        if date not in data_by_date:
            data_by_date[date] = {}
        data_by_date[date][item] = np.array(values, dtype=float)
    date_mats = {}
    for date, m in data_by_date.items():
        mat = np.zeros((len(FEATURE_NAMES), 24))
        for i, feat in enumerate(FEATURE_NAMES):
            mat[i, :] = m.get(feat, 0.0)
        date_mats[date] = mat
    return date_mats

# ----------------------------
# 构造滑动窗口样本
# ----------------------------
def build_samples(date_mats, window=9):
    dates = sorted(date_mats.keys())
    mats = [date_mats[d] for d in dates]
    full = np.hstack(mats)
    N_hours = full.shape[1]
    X_list, y_list = [], []
    pm25_idx = FEATURE_NAMES.index(TARGET_FEATURE)
    for t in range(N_hours - window):
        X_list.append(full[:, t:t+window].flatten())
        y_list.append(full[pm25_idx, t+window])
    X = np.vstack(X_list)
    y = np.array(y_list).reshape(-1,1)
    return X, y

# ----------------------------
# 缺失值填充和标准化
# ----------------------------
def preprocess_fill_and_normalize(X):
    X = X.copy().astype(float)
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_adj = np.where(std==0, 1.0, std)
    X_norm = (X - mean)/std_adj
    return X_norm, mean, std_adj

# ----------------------------
# Adagrad 训练
# ----------------------------
def adagrad_train(X, y, lr=LR, iteration=ITERATION, eps=EPS, lmbd=LAMBDA,
                  X_val=None, y_val=None, loss_threshold=LOSS_THRESHOLD,
                  patience=PATIENCE, verbose=True):
    N,D = X.shape
    w = np.zeros((D,1))
    G = np.zeros((D,1))
    losses, val_losses = [], []
    best_val_loss = float('inf')
    best_w = None
    no_improve_count = 0

    for it in range(iteration):
        pred = X.dot(w)
        err = pred - y
        if np.any(np.isnan(err)) or np.any(np.isinf(err)):
            print(f"NaN/Inf encountered at iter {it+1}, stopping.")
            break
        loss = np.mean(err**2)/2.0 + lmbd/2*np.sum(w**2)
        grad = (X.T.dot(err)/N) + lmbd*w
        G += grad**2
        ada = np.sqrt(G)+eps
        w = w - lr*grad/ada
        losses.append(loss)

        if X_val is not None and y_val is not None:
            pred_val = X_val.dot(w)
            err_val = pred_val - y_val
            val_loss = np.mean(err_val**2)/2.0 + lmbd/2*np.sum(w**2)
            val_losses.append(val_loss)
            if val_loss < best_val_loss - 1e-12:
                best_val_loss = val_loss
                best_w = w.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            if (loss_threshold is not None) and (val_loss <= loss_threshold):
                print(f"Early stop: val_loss {val_loss:.6e} <= threshold at iter {it+1}")
                break
            if no_improve_count >= patience:
                print(f"Early stop: no improvement for {patience} iters at iter {it+1}")
                break

        if verbose and (it<10 or it%max(1,iteration//10)==0):
            if X_val is not None:
                print(f"iter {it+1}/{iteration} loss={loss:.6f} val={val_losses[-1]:.6f}")
            else:
                print(f"iter {it+1}/{iteration} loss={loss:.6f}")
    final_w = best_w if best_w is not None else w
    return final_w, losses, val_losses

# ----------------------------
# 保存/加载模型
# ----------------------------
def save_model(path, w, mean, std):
    np.savez(path, w=w, mean=mean, std=std)
    print("Model saved:", path)

def load_model(path):
    npz = np.load(path)
    return npz['w'], npz['mean'], npz['std']

# ----------------------------
# 测试集预测
# ----------------------------
def predict_test_csv(test_csv_path, model_path, out_csv=PREDICT_CSV):
    w, mean, std = load_model(model_path)
    df = pd.read_csv(test_csv_path, header=None, encoding='gbk').fillna(0).replace("NR", 0)
    results = []
    for idx in range(0, df.shape[0], 18):
        block = df.iloc[idx:idx+18,:]
        if block.shape[0]<18: break
        sample_id = block.iloc[0,0]
        vals = [block.iloc[r,2:].astype(float).values for r in range(18)]
        feat = np.array(vals).flatten()
        feat = (feat - mean)/std
        feat = np.concatenate(([1.0], feat))
        pred_log = float(np.dot(feat, w).squeeze())
        pred = float(np.expm1(pred_log))
        pred = max(0.0, pred)
        results.append([sample_id, pred])
    out_df = pd.DataFrame(results, columns=['id','value'])
    out_df.to_csv(out_csv, index=False)
    print("Predictions written to", out_csv)
    return out_df

# ----------------------------
# 主流程
# ----------------------------
def main():
    print("Parsing train.csv ...")
    date_mats = parse_train_csv(TRAIN_CSV)
    print("Dates loaded:", len(date_mats))
    X_raw, y = build_samples(date_mats)
    print("Samples:", X_raw.shape)

    y_orig = y.copy()
    y_orig[y_orig<0]=0.0
    y_log = np.log1p(y_orig)

    X_tr, X_val, y_tr_log, y_val_log = train_test_split(X_raw, y_log, test_size=VALIDATION_SPLIT, random_state=42)
    _, _, y_tr_orig, y_val_orig = train_test_split(X_raw, y_orig, test_size=VALIDATION_SPLIT, random_state=42)

    X_tr_norm, mean, std = preprocess_fill_and_normalize(X_tr)
    X_val_norm = (X_val - mean)/std

    X_tr_bias = np.concatenate((np.ones((X_tr_norm.shape[0],1)), X_tr_norm), axis=1)
    X_val_bias = np.concatenate((np.ones((X_val_norm.shape[0],1)), X_val_norm), axis=1)

    print("Start training ...")
    w, losses, val_losses = adagrad_train(X_tr_bias, y_tr_log, X_val=X_val_bias, y_val=y_val_log, lr=LR)

    save_model(MODEL_PATH, w, mean, std)
    np.savez(BEST_MODEL_PATH, w=w, mean=mean, std=std)

    y_tr_pred = np.expm1(X_tr_bias.dot(w))
    y_val_pred = np.expm1(X_val_bias.dot(w))
    y_tr_pred = np.maximum(0.0, y_tr_pred)
    y_val_pred = np.maximum(0.0, y_val_pred)

    def metrics(y_t, y_p):
        return mean_squared_error(y_t, y_p), np.sqrt(mean_squared_error(y_t, y_p)), mean_absolute_error(y_t, y_p), r2_score(y_t, y_p)

    tr_mse, tr_rmse, tr_mae, tr_r2 = metrics(y_tr_orig, y_tr_pred)
    val_mse, val_rmse, val_mae, val_r2 = metrics(y_val_orig, y_val_pred)

    print("--- Training metrics ---")
    print(f"Train MSE={tr_mse:.6f}, RMSE={tr_rmse:.6f}, MAE={tr_mae:.6f}, R2={tr_r2:.6f}")
    print(f"Val   MSE={val_mse:.6f}, RMSE={val_rmse:.6f}, MAE={val_mae:.6f}, R2={val_r2:.6f}")

    # 绘图
    plt.figure(figsize=(8,5))
    plt.plot(losses, label='train loss')
    plt.plot(range(len(val_losses)), val_losses, label='val loss')
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log-space)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved training loss plot")

    if os.path.exists(TEST_CSV):
        predict_test_csv(TEST_CSV, MODEL_PATH)
    else:
        print("No test.csv found, skipping test prediction.")

if __name__ == "__main__":
    main()
