# -*- coding: utf-8 -*-
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# データの読み込みと前処理
# CSV読み込み（インデックスを正しく扱うため index_col=0 を指定）
df = pd.read_csv("detaset.csv", index_col=0)

# 目的変数（ラベル）と説明変数（特徴量ベクトル）の切り分け
y = df["label"]
X = df.drop(columns=["label"])

# 訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

# 各種分類モデルの学習と評価（ループでスマートに処理）
models = {
    "Support Vector Machine": SVC(kernel="rbf", random_state=42),
    "Gaussian Naive Bayes": GaussianNB(priors=None, var_smoothing=1e-09),
    "k-NN (k=16)": KNeighborsClassifier(n_neighbors=16),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    print(f"\n{"="*10} {name} Prediction {"="*10}")
    print(f"予測結果: {predicted}")
    print(classification_report(y_test, predicted, digits=5))

# k-NN 近傍数 (k) の変化グラフプロット

accuracy_list = []
k_range = range(1, 100)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test) 
    accuracy_list.append(metrics.accuracy_score(y_test, Y_pred))

plt.figure(figsize=(8, 5))
plt.plot(k_range, accuracy_list, marker="o", markersize=3, linestyle="-")
plt.xlabel("k-NN (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("k-NN Accuracy by Hyperparameter k")
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("knn_result.png", dpi=300)
plt.show()

# PCAによる2次元への次元削減

pca = PCA(n_components=2, random_state=42)

# 訓練データの基準（主成分空間）を元にして、テストデータを変換（transform）
pca_X_train = pca.fit_transform(X_train)
pca_X_test = pca.transform(X_test)

# 次元削減後のデータでSVMを再学習
svm_plot = SVC(kernel="rbf", random_state=42)
svm_plot.fit(pca_X_train, y_train)

svm_plot_predicted_train = svm_plot.predict(pca_X_train)
svm_plot_predicted_test = svm_plot.predict(pca_X_test)

print(f"\n{"="*10} SVM (After PCA) Train Report {"="*10}")
print(classification_report(y_train, svm_plot_predicted_train, digits=5))
print(f"\n{"="*10} SVM (After PCA) Test Report {"="*10}")
print(classification_report(y_test, svm_plot_predicted_test, digits=5))

# SVM決定境界の可視化プロット
gridsize = 0.02  # 境界線を滑らかに描写するために細かく調整
margin = 0.5
x_min, x_max = pca_X_train[:, 0].min() - margin, pca_X_train[:, 0].max() + margin
y_min, y_max = pca_X_train[:, 1].min() - margin, pca_X_train[:, 1].max() + margin
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, gridsize), np.arange(y_min, y_max, gridsize)
)

# 決定境界の予測
Z = svm_plot.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 8))
# 境界線の塗りつぶし表示
plt.contourf(xx, yy, Z, cmap=plt.cm.terrain, alpha=0.3)

# 各クラスの散布図プロット（クラス数やラベル名に自動で対応できるよう柔軟化）
unique_labels = sorted(y_train.unique())
colors = cm.rainbow(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = y_train == label
    plt.scatter(
        pca_X_train[mask, 0],
        pca_X_train[mask, 1],
        color=color,
        label=f"Class {label}",
        edgecolors="k",
        alpha=0.8,
    )

# グラフの詳細設定
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="best")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("SVM Decision Boundary with PCA-reduced Data")
plt.savefig("pca_result.png", dpi=300, bbox_inches="tight")
plt.show()