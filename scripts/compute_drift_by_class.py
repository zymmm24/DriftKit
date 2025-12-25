import os
import numpy as np
import csv
from scipy.linalg import inv

# ================= 配置 =================
BASELINE_PATH = "outputs/baseline_stats.npz"
VAL_FEATURE_DIR = "outputs/val"
OUT_CSV = "outputs/val_drift_scores.csv"
EPS = 1e-6
# =======================================


def mahalanobis_distance(x, mean, cov_inv):
    """
    x:    [D]
    mean: [D]
    cov_inv: [D, D]
    """
    diff = x - mean
    return float(np.sqrt(diff.T @ cov_inv @ diff))


print("== Computing drift scores (VAL vs BASELINE) ==")

# ========= Load baseline =========
baseline = np.load(BASELINE_PATH)

baseline_means = {}
baseline_cov_inv = {}

for k in baseline.files:
    if k.endswith("_mean"):
        class_id = k.split("_")[1]
        baseline_means[class_id] = baseline[k]
    elif k.endswith("_cov"):
        class_id = k.split("_")[1]
        cov = baseline[k]
        cov += EPS * np.eye(cov.shape[0])  # 数值稳定
        baseline_cov_inv[class_id] = inv(cov)

print(f"Loaded baseline for classes: {sorted(baseline_means.keys())}")

# ========= Prepare output =========
rows = []

# ========= Process VAL features =========
for fname in sorted(os.listdir(VAL_FEATURE_DIR)):
    if not fname.endswith("_features.npy"):
        continue

    # 文件名示例：1_features.npy
    class_id = fname.replace("_features.npy", "")
    path = os.path.join(VAL_FEATURE_DIR, fname)

    if class_id not in baseline_means:
        print(f"⚠️ Skip class {class_id} (no baseline)")
        continue

    X = np.load(path)  # [N, D]
    mean = baseline_means[class_id]
    cov_inv = baseline_cov_inv[class_id]

    print(f"\nClass {class_id}: {X.shape[0]} samples")

    for i, x in enumerate(X):
        score = mahalanobis_distance(x, mean, cov_inv)
        rows.append({
            "class": class_id,
            "sample_id": i,
            "drift_score": score
        })

# ========= Save CSV =========
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["class", "sample_id", "drift_score"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved drift scores:", OUT_CSV)

# ========= Simple statistics =========
scores = np.array([r["drift_score"] for r in rows], dtype=np.float32)

print("\n===== Drift Score Summary (VAL) =====")
print(f"Total samples: {len(scores)}")
print(f"Mean drift:   {scores.mean():.4f}")
print(f"Std drift:    {scores.std():.4f}")
print(f"Max drift:    {scores.max():.4f}")
print(f"95% quantile: {np.percentile(scores, 95):.4f}")

print("\n🎯 DRIFT COMPUTATION DONE")
