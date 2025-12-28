#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_features.py
功能：可视化 train.csv / test.csv 中每个特征的分布（连续 & 离散）。
输出：把所有图片与统计表保存到 visualization_outputs/ 子目录中。
使用：python visualize_features.py --train train.csv --test test.csv
"""

import os
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 配置（可按需修改）
# ----------------------------
DEFAULT_OUTPUT_DIR = "visualization_outputs"
CONTINUOUS_FEATURES = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]


# 推断离散特征时会以 train.csv 为准
# ----------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def detect_discrete_columns(df, continuous_list):
    """自动检测离散特征：除去连续特征与 label 列外的列视为离散（字符串或类别型）。"""
    candidates = [c for c in df.columns if c not in continuous_list and c != "income"]
    return candidates


def plot_continuous(df_train, df_test, col, outdir):
    """画连续变量的直方图、箱线图、训练/测试叠加密度图等"""
    ensure_dir(outdir)
    plt.figure(figsize=(10, 6))
    # 训练集直方图
    sns.histplot(df_train[col].dropna(), kde=False, stat="density", label="train", alpha=0.6)
    # 测试集直方图
    if col in df_test.columns:
        sns.histplot(df_test[col].dropna(), kde=False, stat="density", label="test", alpha=0.6)
    plt.title(f"{col} - Histogram (train vs test)")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{col}_hist.png"))
    plt.close()

    # KDE (密度估计)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df_train[col].dropna(), label="train")
    if col in df_test.columns:
        sns.kdeplot(df_test[col].dropna(), label="test")
    plt.title(f"{col} - KDE")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"{col}_kde.png"))
    plt.close()

    # 箱线图（train 按 income 分组）
    if "income" in df_train.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="income", y=col, data=df_train)
        plt.title(f"{col} - Boxplot by income (train)")
        plt.savefig(os.path.join(outdir, f"{col}_box_by_income.png"))
        plt.close()

    # ================================
    # 新增：小提琴图（violin plot）+ swarmplot（柄图效果）
    # ================================
    if "income" in df_train.columns:
        plt.figure(figsize=(8, 6))

        # 随机采样以避免 swarmplot 挤不下（保持可视化效果）
        df_sample = df_train.sample(n=min(2000, len(df_train)), random_state=42)

        sns.violinplot(x="income", y=col, data=df_train, inner=None, color="lightgray")
        sns.boxplot(x="income", y=col, data=df_train, showfliers=False)

        # 对采样后的数据画 swarmplot
        sns.swarmplot(x="income", y=col, data=df_sample, size=3, alpha=0.6)

        plt.title(f"{col} - Violin + Swarm (train)")
        plt.savefig(os.path.join(outdir, f"{col}_violin_swarm.png"))
        plt.close()


def plot_discrete(df_train, df_test, col, outdir, top_k=20):
    """画离散变量的柱状图（计数），并保存类别分布表"""
    ensure_dir(outdir)
    train_counts = df_train[col].fillna("NA").astype(str).value_counts()
    test_counts = df_test[col].fillna("NA").astype(str).value_counts() if col in df_test.columns else pd.Series()
    # 合并 top_k
    top = train_counts.index.tolist()[:top_k]
    plt.figure(figsize=(12, 6))
    sns.barplot(x=train_counts.index[:top_k], y=train_counts.values[:top_k])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{col} - Top {top_k} categories (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{col}_top{top_k}_train.png"))
    plt.close()

    # 保存分布表
    dist_df = pd.DataFrame({
        "category": train_counts.index,
        "train_count": train_counts.values,
        "test_count": [test_counts.get(c, 0) for c in train_counts.index]
    })
    dist_df.to_csv(os.path.join(outdir, f"{col}_distribution.csv"), index=False)

    # 如果类别过多，绘制累计条形图（前50）
    if len(train_counts) > top_k:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=train_counts.index[:50], y=train_counts.values[:50])
        plt.xticks(rotation=90)
        plt.title(f"{col} - Top 50 categories (train)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}_top50_train.png"))
        plt.close()

    # ================================
    # 新增：离散特征的饼图（Pie Chart）
    # ================================
    plt.figure(figsize=(8, 8))
    train_counts[:top_k].plot(kind="pie", autopct="%1.1f%%", startangle=90)
    plt.ylabel("")
    plt.title(f"{col} - Pie Chart (Top {top_k}, train)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{col}_pie_top{top_k}.png"))
    plt.close()

    # ================================
    # 新增：横向条形图（Horizontal Bar Plot）
    # ================================
    plt.figure(figsize=(10, 6))
    sns.barplot(y=train_counts.index[:top_k], x=train_counts.values[:top_k], orient="h")
    plt.title(f"{col} - Top {top_k} Categories (Horizontal, train)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{col}_hbar_top{top_k}.png"))
    plt.close()



def main(args):
    train_path = args.train
    test_path = args.test
    outdir = args.outdir

    ensure_dir(outdir)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. 列出列名 & 基本统计
    summary = {}
    for c in df_train.columns:
        summary[c] = {
            "dtype": str(df_train[c].dtype),
            "n_unique": int(df_train[c].nunique()),
            "n_missing": int(df_train[c].isna().sum())
        }
    pd.DataFrame.from_dict(summary, orient="index").to_csv(os.path.join(outdir, "columns_summary.csv"))

    # 2. 连续 & 离散特征处理
    continuous = [c for c in CONTINUOUS_FEATURES if c in df_train.columns]
    discrete = detect_discrete_columns(df_train, continuous)

    # 3. 绘图
    cont_dir = os.path.join(outdir, "continuous")
    disc_dir = os.path.join(outdir, "discrete")
    ensure_dir(cont_dir);
    ensure_dir(disc_dir)

    for col in continuous:
        try:
            plot_continuous(df_train, df_test, col, cont_dir)
        except Exception as e:
            print(f"绘制连续变量 {col} 出错：{e}")

    for col in discrete:
        try:
            plot_discrete(df_train, df_test, col, disc_dir)
        except Exception as e:
            print(f"绘制离散变量 {col} 出错：{e}")

    print("可视化已完成，输出目录：", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv", help="训练数据 csv 路径")
    parser.add_argument("--test", type=str, default="test.csv", help="测试数据 csv 路径")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTPUT_DIR, help="可视化输出文件夹")
    args = parser.parse_args()
    main(args)
