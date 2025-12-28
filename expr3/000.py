import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.manifold import TSNE
import seaborn as sns


# ============================================================
# Logistic Regression 手写实现（含 loss 记录）
# ============================================================
class LogisticRegressionGD:

    def __init__(self, lr=0.01, epochs=2000):
        self.lr = lr
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        eps = 1e-12
        return -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))

    def fit(self, X, y):
        N, D = X.shape
        self.w = np.zeros(D)

        for i in range(self.epochs):
            z = X.dot(self.w)
            y_pred = self.sigmoid(z)

            grad = X.T.dot(y_pred - y) / N
            self.w -= self.lr * grad

            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            if i % 200 == 0:
                print(f"[Train] Epoch {i}, Loss = {loss:.6f}")

    def predict_proba(self, X):
        return self.sigmoid(X.dot(self.w))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


# ============================================================
# 数据预处理函数
# ============================================================
def preprocess(df_train, df_test, z_norm_cols):

    # 你给的特征
    cont_cols = [
        "age", "fnlwgt", "education-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    disc_cols = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "sex", "native-country"
    ]

    # 1. 处理连续特征
    X_cont_train = df_train[cont_cols].values.astype(float)
    X_cont_test = df_test[cont_cols].values.astype(float)

    # ---- 正态分布规格化（第二种方式） ----
    mean = X_cont_train.mean(axis=0)
    std = X_cont_train.std(axis=0) + 1e-12
    X_cont_train = (X_cont_train - mean) / std
    X_cont_test = (X_cont_test - mean) / std

    # 2. 处理离散特征（One-Hot）
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_disc_train = ohe.fit_transform(df_train[disc_cols])
    X_disc_test = ohe.transform(df_test[disc_cols])

    # 保存 one-hot 后的特征名称
    feature_names_cont = cont_cols
    feature_names_disc = ohe.get_feature_names_out(disc_cols).tolist()
    feature_names = feature_names_cont + feature_names_disc

    # 拼接最终矩阵
    X_train = np.hstack([X_cont_train, X_disc_train])
    X_test = np.hstack([X_cont_test, X_disc_test])

    print(f"最终特征维度: {X_train.shape[1]}")

    return X_train, X_test, feature_names


# ============================================================
# 主训练流程
# ============================================================
def main():

    # =======================
    # 1. 读数据
    # =======================
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")

    print(df_train.columns.tolist())

    y_train = (df_train["income"] == " >50K").astype(int).values

    # 指定正态规格化要应用于哪些列（例如 One-Hot 后的索引）
    z_norm_cols = [0, 1, 3, 4, 5, 7, 10, 12, 25, 26, 27, 28]

    X_train, X_test, feature_names = preprocess(df_train, df_test, z_norm_cols)

    # =======================
    # 2. 训练模型
    # =======================
    model = LogisticRegressionGD(lr=0.05, epochs=2000)
    model.fit(X_train, y_train)

    # =======================
    # 3. 预测
    # =======================
    y_pred_prob = model.predict_proba(X_train)
    y_pred = model.predict(X_train)

    # =======================
    # 4. 输出评估指标
    # =======================
    acc = accuracy_score(y_train, y_pred)
    auc = roc_auc_score(y_train, y_pred_prob)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_prob))
    r2 = r2_score(y_train, y_pred_prob)

    print("\n===== 训练集表现 =====")
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("RMSE:", rmse)
    print("R2:", r2)

    # =======================
    # 5. loss 曲线
    # =======================
    plt.figure(figsize=(7,5))
    plt.plot(model.loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()

    # =======================
    # 6. 特征权重排序（特征贡献度）
    # =======================
    w = model.w
    ind = np.argsort(np.abs(w))[::-1]

    print("\n===== 特征贡献度排序（前20） =====")
    for i in ind[:20]:
        print(f"{feature_names[i]:25s}  weight = {w[i]:.6f}")

    # =======================
    # 7. t-SNE 可视化
    # =======================
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    X_embed = tsne.fit_transform(X_train)

    plt.figure(figsize=(7,6))
    sns.scatterplot(x=X_embed[:,0], y=X_embed[:,1], hue=y_train, palette="coolwarm", s=10)
    plt.title("t-SNE Visualization")
    plt.savefig("tsne.png")
    plt.close()

    print("\n可视化已保存：loss_curve.png, tsne.png")


if __name__ == "__main__":
    main()
