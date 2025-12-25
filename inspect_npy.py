import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 避免 PyCharm backend 报错

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ======================
# 0. Sanity: working dir
# ======================
print("CWD:", os.getcwd())

# ======================
# 1. Load features
# ======================
# ⚠️ 使用真实存在、样本数足够的文件
feature_path = "scripts/outputs/train_1_features.npy"

if not os.path.exists(feature_path):
    raise FileNotFoundError(f"Feature file not found: {feature_path}")

X = np.load(feature_path)

print("Loaded:", feature_path)
print("Shape:", X.shape)          # (N, 256) —— N 是“目标数”
print("Dtype:", X.dtype)
print("Mean / Std:", X.mean(), X.std())

# ======================
# 2. Sanity check
# ======================
if X.ndim != 2:
    raise ValueError(f"Expected 2D array, got shape {X.shape}")

if X.shape[0] < 10:
    raise ValueError(
        f"Too few samples for PCA: {X.shape[0]}. "
        "Need at least 10 targets."
    )

# ======================
# 3. PCA
# ======================
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# ======================
# 4. Visualization
# ======================
plt.figure(figsize=(6, 5))
plt.scatter(X2[:, 0], X2[:, 1], s=8, alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Baseline Feature Distribution (Class 1, PCA)")
plt.tight_layout()

out_path = "baseline_pca_train1.png"
plt.savefig(out_path, dpi=150)
plt.close()

print(f"Saved: {out_path}")
