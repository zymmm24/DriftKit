import os
import numpy as np
import csv

# ================= 配置 =================
TRAIN_FEATURE_DIR = "outputs/train"   # ✅ 只用 train
OUT_NPZ = "outputs/baseline_stats.npz"
OUT_CSV = "outputs/baseline_summary.csv"
EPS = 1e-6
# =======================================


def compute_mean_cov(X, eps=1e-6):
    """
    X: [N, D]
    return:
      mean: [D]
      cov : [D, D]
    """
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += eps * np.eye(cov.shape[0])  # 数值稳定
    return mean.astype(np.float32), cov.astype(np.float32)


baseline = {}
summary_rows = []

print("== Building BASELINE statistics (TRAIN only) ==")

if not os.path.isdir(TRAIN_FEATURE_DIR):
    raise FileNotFoundError(f"Not found: {TRAIN_FEATURE_DIR}")

for fname in sorted(os.listdir(TRAIN_FEATURE_DIR)):
    if not fname.endswith("_features.npy"):
        continue

    # 文件名示例：1_features.npy
    class_id = fname.replace("_features.npy", "")
    path = os.path.join(TRAIN_FEATURE_DIR, fname)

    X = np.load(path)
    N, D = X.shape

    print(f"\nClass {class_id}: {N} samples")

    if N < 10:
        print("  ⚠️ Skip (too few samples)")
        continue

    mean, cov = compute_mean_cov(X, EPS)

    # ===== 机器可用 =====
    baseline[f"class_{class_id}_mean"] = mean
    baseline[f"class_{class_id}_cov"] = cov

    # ===== 人可读统计 =====
    feature_var = np.diag(cov)

    summary_rows.append({
        "class": class_id,
        "num_samples": N,
        "feature_dim": D,
        "mean_mean": float(mean.mean()),
        "mean_std": float(mean.std()),
        "total_variance_trace": float(feature_var.sum()),
        "max_feature_variance": float(feature_var.max()),
        "min_feature_variance": float(feature_var.min())
    })

# ================= 保存 =================

# 机器可读（后续漂移计算用）
np.savez(OUT_NPZ, **baseline)
print("\nSaved baseline stats (npz):", OUT_NPZ)

# 人可读（你 / 导师 / 报告用）
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print("Saved baseline summary (csv):", OUT_CSV)
print("\n🎯 BASELINE BUILD DONE (TRAIN ONLY)")
