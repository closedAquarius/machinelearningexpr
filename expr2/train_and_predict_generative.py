#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_and_predict_generative.py
功能：对给定 train.csv 进行概率生成模型训练（多元高斯类条件分布），对 test.csv 做预测并输出 predict.csv。
支持模型类型：naive (高斯朴素贝叶斯, 对角协方差)、lda (线性判别-共享协方差)、qda (二类各自协方差)。
输出目录：model_outputs/
使用：python train_and_predict_generative.py --train train.csv --test test.csv
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, mean_squared_error, r2_score, accuracy_score

# ----------------------------
# 配置（可调参数集中在这里）
# ----------------------------
config = {
    "model_type": "lda",  # 'naive' / 'lda' / 'qda'
    "scale_continuous": True,  # 是否对连续变量做标准化（0均值1方差）
    "log_transform_gain": True,  # 是否对 capital_gain/loss 做 log1p 变换
    "target_dim": 108,  # 强制 X 的维度（不足补零，多于截断）
    "random_seed": 42,
    "output_dir": "model_outputs",
    "verbose": True
}
# ----------------------------

np.random.seed(config["random_seed"])

# 列定义（与题目一致）
CONTINUOUS_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
DISCRETE_CANDIDATES = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex",
                       "native_country"]
ALL_COLUMNS = CONTINUOUS_FEATURES + DISCRETE_CANDIDATES + ["income"]


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# ----------------------------
# 数据预处理模块
# ----------------------------
def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def preprocess(df_train, df_test, config):
    """
    1. 分离连续与离散特征
    2. 对离散特征用 OneHotEncoder(fit on train, transform both)
    3. 对连续特征可选择标准化 / log transform
    4. 合并成 X_train, X_test；同时生成 y_train (0/1)
    5. 强制调整维度为 config['target_dim']（pad/trim）
    """
    out = {}
    # 确保目录
    ensure_dir(config["output_dir"])

    # 1. 连续
    cont = [c for c in CONTINUOUS_FEATURES if c in df_train.columns]
    X_cont_train = df_train[cont].copy()
    X_cont_test = df_test[cont].copy()

    # 可选 log1p
    if config.get("log_transform_gain", False):
        for c in ["capital_gain", "capital_loss"]:
            if c in X_cont_train.columns:
                X_cont_train[c] = np.log1p(X_cont_train[c].fillna(0))
                X_cont_test[c] = np.log1p(X_cont_test[c].fillna(0))

    # 标准化（fit on train）
    scaler = None
    if config.get("scale_continuous", True):
        scaler = StandardScaler()
        X_cont_train = pd.DataFrame(scaler.fit_transform(X_cont_train.fillna(0)), columns=cont)
        X_cont_test = pd.DataFrame(scaler.transform(X_cont_test.fillna(0)), columns=cont)
    else:
        X_cont_train = X_cont_train.fillna(0)
        X_cont_test = X_cont_test.fillna(0)

    # 2. 离散 OneHot（fit on train）
    disc = [c for c in DISCRETE_CANDIDATES if c in df_train.columns]
    # 将缺失替换为字符串 'NA' 以便编码器识别为类别
    df_train_disc = df_train[disc].fillna("NA").astype(str)
    df_test_disc = df_test[disc].fillna("NA").astype(str)

    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_disc_train = ohe.fit_transform(df_train_disc)
    X_disc_test = ohe.transform(df_test_disc)

    show_onehot_vectors(ohe, df_train_disc.columns)

    # 合并
    X_train = np.hstack([X_cont_train.values, X_disc_train])
    X_test = np.hstack([X_cont_test.values, X_disc_test])

    # 处理 label
    y_map = {" <=50K": 0, " <=50K.": 0, " >50K": 1, " >50K.": 1}  # 兼容可能的格式
    if "income" in df_train.columns:
        y_train = df_train["income"].map(y_map).astype(int).values
    else:
        raise ValueError("train.csv 必须包含 income 列")

    # 强制维度为 target_dim（padding/trimming）
    def enforce_dim(X, target_dim):
        n, d = X.shape
        if d == target_dim:
            return X
        elif d < target_dim:
            pad = np.zeros((n, target_dim - d))
            return np.hstack([X, pad])
        else:
            # 稍微警告并截断
            print("截断")
            return X[:, :target_dim]

    X_train = enforce_dim(X_train, config["target_dim"])
    X_test = enforce_dim(X_test, config["target_dim"])

    # 保存一些预处理对象与信息
    out["X_train"] = X_train
    out["X_test"] = X_test
    out["y_train"] = y_train
    out["ohe"] = ohe
    out["scaler"] = scaler
    out["cont_columns"] = cont
    out["disc_columns"] = ohe.get_feature_names_out(disc).tolist()
    out["preprocess_info"] = {
        "original_disc_columns": disc,
        "ohe_feature_names": out["disc_columns"],
        "final_feature_dim": X_train.shape[1]
    }
    return out


# 打印编码结果
def show_onehot_vectors(encoder, feature_names):
    categories = encoder.categories_

    print("\n======= 每个特征的类别 → One-Hot 向量 =======")

    start = 0  # 每个特征在 One-Hot 向量中的起始列
    for feat, cats in zip(feature_names, categories):
        k = len(cats)  # 当前特征的类别数量
        print(f"\n【特征：{feat}】 (类别数量 {k})")

        for i, cat in enumerate(cats):
            vec = np.zeros(k, dtype=int)
            vec[i] = 1
            print(f"  类别: {cat}  →  {vec.tolist()}")

        start += k


# ----------------------------
# 概率生成模型实现（闭式解）
# ----------------------------
class GaussianGenerativeModel:
    """
    实现三种模式：
    - naive: 假设条件独立（对角协方差，即各维独立高斯），实现类似 GaussianNB
    - lda: 假设每类同高斯但共享协方差（线性判别分析 -> 类似生成式线性分类）
    - qda: 每类独立协方差（二次判别分析）
    """

    def __init__(self, model_type="lda", verbose=True):
        assert model_type in ("naive", "lda", "qda")
        self.model_type = model_type
        self.verbose = verbose
        # parameters to learn
        self.phi = None  # class prior P(y=1)
        self.mu0 = None
        self.mu1 = None
        self.sigma_shared = None
        self.sigma0 = None
        self.sigma1 = None
        self.eps = 1e-8  # 数值稳定项

    def fit(self, X, y):
        n, d = X.shape
        # priors
        phi = np.mean(y == 1)
        x0 = X[y == 0]
        x1 = X[y == 1]
        mu0 = np.mean(x0, axis=0)
        mu1 = np.mean(x1, axis=0)

        # 计算协方差
        if self.model_type == "naive":
            # 对角协方差：只取每维方差（加偏置 eps）
            var0 = np.var(x0, axis=0) + self.eps
            var1 = np.var(x1, axis=0) + self.eps
            self.var0 = var0
            self.var1 = var1
        elif self.model_type == "lda":
            # 共享协方差（无偏估计）
            # sigma = ((n0-1)*cov0 + (n1-1)*cov1) / (n0+n1-2)
            cov0 = np.cov(x0, rowvar=False)
            cov1 = np.cov(x1, rowvar=False)
            pooled = ((x0.shape[0] - 1) * cov0 + (x1.shape[0] - 1) * cov1) / (n - 2)
            # 添加小常数确保可逆
            pooled += np.eye(d) * self.eps
            self.sigma_shared = pooled
        elif self.model_type == "qda":
            cov0 = np.cov(x0, rowvar=False) + np.eye(d) * self.eps
            cov1 = np.cov(x1, rowvar=False) + np.eye(d) * self.eps
            self.sigma0 = cov0
            self.sigma1 = cov1

        self.phi = phi
        self.mu0 = mu0
        self.mu1 = mu1

        if self.verbose:
            print("模型类型:", self.model_type)
            print("phi (P(y=1)):", self.phi)
            print("mu0 shape:", self.mu0.shape, "mu1 shape:", self.mu1.shape)
            if self.model_type == "naive":
                print("var0/var1 示例前10维:", self.var0[:10], self.var1[:10])
            elif self.model_type == "lda":
                print("sigma_shared shape:", self.sigma_shared.shape)
            elif self.model_type == "qda":
                print("sigma0/sigma1 shapes:", self.sigma0.shape, self.sigma1.shape)

    def _log_pdf_multivariate(self, x, mu, sigma):
        """
        计算单点 x 在 N(mu, sigma) 下的 log p(x)
        sigma 为协方差矩阵（dxd）
        """
        d = x.shape[0]
        # 使用 np.linalg.slogdet + solve 保持数值稳定
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            # 避免奇异
            sigma = sigma + np.eye(d) * self.eps
            sign, logdet = np.linalg.slogdet(sigma)
        diff = x - mu
        inv = np.linalg.inv(sigma)
        val = -0.5 * (diff.T @ inv @ diff) - 0.5 * (d * np.log(2 * np.pi) + logdet)
        return val

    def predict_proba(self, X):
        """
        返回 P(y=1 | x) 的概率（数组）
        对于每个样本计算 p(x|y=1)*phi / (p(x|y=1)*phi + p(x|y=0)*(1-phi))
        """
        n, d = X.shape
        log_px_y0 = np.zeros(n)
        log_px_y1 = np.zeros(n)

        if self.model_type == "naive":
            # 对角协方差，逐维计算
            # log p(x|y) = sum_k ( -0.5*log(2pi var) - (x_k-mu_k)^2/(2 var_k) )
            var0 = self.var0
            var1 = self.var1
            const0 = -0.5 * np.sum(np.log(2 * np.pi * var0))
            const1 = -0.5 * np.sum(np.log(2 * np.pi * var1))
            # 广播计算
            diffs0 = (X - self.mu0) ** 2
            diffs1 = (X - self.mu1) ** 2
            log_px_y0 = const0 - 0.5 * np.sum(diffs0 / var0, axis=1)
            log_px_y1 = const1 - 0.5 * np.sum(diffs1 / var1, axis=1)
        elif self.model_type == "lda":
            # 使用多元高斯的 logpdf，sigma_shared
            inv = np.linalg.inv(self.sigma_shared)
            sign, logdet = np.linalg.slogdet(self.sigma_shared)
            const = -0.5 * (d * np.log(2 * np.pi) + logdet)
            # 逐样本
            for i in range(n):
                diff0 = X[i] - self.mu0
                diff1 = X[i] - self.mu1
                log_px_y0[i] = const - 0.5 * (diff0.T @ inv @ diff0)
                log_px_y1[i] = const - 0.5 * (diff1.T @ inv @ diff1)
        elif self.model_type == "qda":
            for i in range(n):
                log_px_y0[i] = self._log_pdf_multivariate(X[i], self.mu0, self.sigma0)
                log_px_y1[i] = self._log_pdf_multivariate(X[i], self.mu1, self.sigma1)
        # 合并先验
        log_prior1 = np.log(self.phi + 1e-12)
        log_prior0 = np.log(1 - self.phi + 1e-12)
        # log numerator and denominator (log-sum-exp)
        log_num = log_px_y1 + log_prior1
        log_den = np.vstack([log_px_y0 + log_prior0, log_px_y1 + log_prior1])
        # logsumexp
        maxv = np.max(log_den, axis=0)
        log_den_sum = maxv + np.log(np.sum(np.exp(log_den - maxv), axis=0))
        log_p1_given_x = log_num - log_den_sum
        p1 = np.exp(log_p1_given_x)
        return p1

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int), proba


# ----------------------------
# 评估与可视化
# ----------------------------
def evaluate_and_visualize(model, X_train, y_train, outdir):
    ensure_dir(outdir)
    # 在训练集上计算概率与预测
    y_pred_label, y_proba = model.predict(X_train)
    acc = accuracy_score(y_train, y_pred_label)
    auc = roc_auc_score(y_train, y_proba)
    r2 = r2_score(y_train, y_proba)  # 把概率当连续值计算 R2
    rmse = np.sqrt(mean_squared_error(y_train, y_proba))

    print("训练集 Accuracy:", acc)
    print("训练集 AUC:", auc)
    print("训练集 R2 (on probabilities):", r2)
    print("训练集 RMSE (on probabilities):", rmse)

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_train, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR");
    plt.ylabel("TPR");
    plt.title("ROC Curve (train)")
    plt.legend()
    plt.savefig(os.path.join(outdir, "roc_train.png"))
    plt.close()

    # 混淆矩阵（基于阈值 0.5）
    cm = confusion_matrix(y_train, y_pred_label)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("pred");
    plt.ylabel("true");
    plt.title("Confusion Matrix (train)")
    plt.savefig(os.path.join(outdir, "confusion_matrix_train.png"))
    plt.close()

    # 保存指标
    with open(os.path.join(outdir, "train_metrics.json"), "w") as f:
        json.dump({"accuracy": float(acc), "auc": float(auc), "r2": float(r2), "rmse": float(rmse)}, f, indent=2)

    # 可视化学到的参数（均值与协方差热图）
    # mu heatmap
    mus = np.vstack([model.mu0, model.mu1])
    plt.figure(figsize=(10, 3))
    sns.heatmap(mus, annot=False, cmap="coolwarm")
    plt.yticks([0.5, 1.5], ["mu0", "mu1"])
    plt.title("Learned class means (mu0, mu1)")
    plt.savefig(os.path.join(outdir, "mus_heatmap.png"))
    plt.close()

    if model.model_type == "lda" and model.sigma_shared is not None:
        plt.figure(figsize=(6, 6))
        sns.heatmap(model.sigma_shared, cmap="viridis")
        plt.title("Shared covariance (sigma_shared)")
        plt.savefig(os.path.join(outdir, "sigma_shared_heatmap.png"))
        plt.close()
    elif model.model_type == "qda":
        plt.figure(figsize=(6, 6))
        sns.heatmap(model.sigma0, cmap="viridis")
        plt.title("Covariance sigma0 (class 0)")
        plt.savefig(os.path.join(outdir, "sigma0_heatmap.png"))
        plt.close()

        plt.figure(figsize=(6, 6))
        sns.heatmap(model.sigma1, cmap="viridis")
        plt.title("Covariance sigma1 (class 1)")
        plt.savefig(os.path.join(outdir, "sigma1_heatmap.png"))
        plt.close()

    return {"accuracy": acc, "auc": auc, "r2": r2, "rmse": rmse}


# ----------------------------
# 主流程
# ----------------------------
def main(args):
    cfg = config.copy()
    cfg["output_dir"] = args.outdir
    ensure_dir(cfg["output_dir"])

    print("加载数据...")
    df_train, df_test = load_data(args.train, args.test)
    print("预处理并编码...")
    prep = preprocess(df_train, df_test, cfg)
    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]

    print("最终特征维度:", X_train.shape[1])
    if X_train.shape[1] != cfg["target_dim"]:
        print(f"警告：最终维度 {X_train.shape[1]} 不等于 target_dim {cfg['target_dim']}，请检查")

    # 训练模型
    model = GaussianGenerativeModel(model_type=cfg["model_type"], verbose=True)
    model.fit(X_train, y_train)

    # 评估并可视化
    eval_dir = os.path.join(cfg["output_dir"], "eval")
    metrics = evaluate_and_visualize(model, X_train, y_train, eval_dir)

    # 对 test 预测并写入 predict.csv
    print("对 test 集进行预测并写入 CSV ...")
    proba_test = model.predict_proba(X_test)
    labels = (proba_test >= 0.5).astype(int)

    # 输出格式：第一行 id,label；id 从 1 到 N (或保持 test 中已有 id，如果存在)
    out_df = pd.DataFrame({
        "id": np.arange(1, len(labels) + 1),
        "label": labels
    })
    predict_path = os.path.join(cfg["output_dir"], "predict.csv")
    out_df.to_csv(predict_path, index=False)
    print("预测已保存：", predict_path)

    # 保存一些模型参数与预处理信息（便于复现）
    model_info = {
        "config": cfg,
        "preprocess_info": prep["preprocess_info"],
        "phi": float(model.phi),
        "model_type": model.model_type,
        "mu0_mean": model.mu0.tolist(),
        "mu1_mean": model.mu1.tolist()
    }
    with open(os.path.join(cfg["output_dir"], "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print("全部完成，输出目录：", cfg["output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="训练数据 csv")
    parser.add_argument("--test", type=str, default="test.csv", help="测试数据 csv")
    parser.add_argument("--outdir", type=str, default=config["output_dir"], help="模型输出目录")
    args = parser.parse_args()
    main(args)
