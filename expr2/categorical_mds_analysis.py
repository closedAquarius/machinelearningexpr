#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
categorical_mds_analysis.py
从 Adult train.csv 中抽取离散特征：
  workclass, education, marital_status, occupation, relationship, race, sex, native_country
对每个离散特征：
  1. One-Hot 编码
  2. 计算类别中心
  3. 使用 MDS(2D) 进行降维可视化
  4. 按 income 上色
  5. 使用卡方检验分析相关性
  6. 🔥（新增）列联表热图 Heatmap
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import MDS
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import chi2_contingency

# ---------------------------
# 配置
# ---------------------------
OUTPUT_DIR = "categorical_outputs"
DISCRETE_CANDIDATES = [
    "workclass","education","marital_status",
    "occupation","relationship","race","sex","native_country"
]

INCOME_MAP = {"<=50K":0, "<=50K.":0, ">50K":1, ">50K.":1}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------
# 主流程
# ---------------------------
def analyze_categorical_mds(train_csv):
    print("Loading dataset...")
    df = pd.read_csv(train_csv)

    # income 清洗 → 0/1
    df["income_clean"] = df["income"].astype(str).str.strip()
    df["income_bin"] = df["income_clean"].map(INCOME_MAP)
    df["income_bin"] = df["income_bin"].fillna(0).astype(int)

    df_disc = df[DISCRETE_CANDIDATES].fillna("NA").astype(str)
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    ohe.fit(df_disc)

    feature_names = ohe.get_feature_names_out(DISCRETE_CANDIDATES)

    # 记录各特征 one-hot 的位置范围
    feature_slices = {}
    start = 0
    for c in DISCRETE_CANDIDATES:
        cats = ohe.categories_[DISCRETE_CANDIDATES.index(c)]
        end = start + len(cats)
        feature_slices[c] = (start, end, cats)
        start = end

    X_disc = ohe.transform(df_disc)

    # ---------------------------------------------------------
    # 对每个离散特征做 MDS + 卡方 + 🔥热图
    # ---------------------------------------------------------
    for feat in DISCRETE_CANDIDATES:
        print(f"\n=== Analyzing feature: {feat} ===")
        s, e, categories = feature_slices[feat]
        X_f = X_disc[:, s:e]

        # 求每个类别的 OHE 均值（相当于中心点）
        centers = []
        income_ratio = []

        for i, cat in enumerate(categories):
            idx = (df[feat].astype(str) == cat)
            if idx.sum() == 0:
                centers.append(np.zeros(X_f.shape[1]))
                income_ratio.append(0)
                continue

            centers.append(X_f[idx].mean(axis=0))
            income_ratio.append(df["income_bin"][idx].mean())

        centers = np.array(centers)

        # Cosine 距离更适合 OHE
        from sklearn.metrics.pairwise import cosine_distances
        dist = cosine_distances(centers)

        # MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
        Y = mds.fit_transform(dist)

        # ------------------------------
        # MDS 绘图
        # ------------------------------
        plt.figure(figsize=(7,6))
        sc = plt.scatter(Y[:,0], Y[:,1], c=income_ratio, cmap="viridis", s=150)

        for i, cat in enumerate(categories):
            plt.text(Y[i,0]+0.01, Y[i,1]+0.01, cat[:12], fontsize=9)

        plt.colorbar(sc, label="P(income > 50K)")
        plt.title(f"MDS of categorical feature: {feat}")
        plt.xlabel("dim1"); plt.ylabel("dim2")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_mds.png"), dpi=150)
        plt.close()

        print(f"[Saved] {feat}_mds.png")

        # ------------------------------
        # 卡方检验
        # ------------------------------
        contingency = pd.crosstab(df[feat], df["income_bin"])
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"Chi-square p-value = {p:.3e}")

        # 保存文本统计
        with open(os.path.join(OUTPUT_DIR, f"{feat}_chi_square.txt"), "w") as f:
            f.write(f"Feature: {feat}\nchi2={chi2}\np={p}\ndof={dof}\n")
            f.write("Contingency Table:\n")
            f.write(str(contingency))

        # =====================================
        # 🔥 NEW: 列联表热图（Heatmap）
        # =====================================
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            contingency,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=0.5,
            cbar=True
        )
        plt.title(f"Heatmap of {feat} vs Income")
        plt.xlabel("Income (0=<=50K, 1=>50K)")
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{feat}_heatmap.png"), dpi=150)
        plt.close()

        print(f"[Saved] {feat}_heatmap.png")

    print("\n全部离散特征 MDS + 热图 + 统计分析完成。输出目录：", OUTPUT_DIR)


if __name__ == "__main__":
    analyze_categorical_mds("train.csv")
