import os
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")  # 避免 PyCharm 后端问题
import matplotlib.pyplot as plt

# ===================== 配置 =====================
FEATURE_DIR = "outputs"
BASELINE_STATS_FILE = os.path.join(FEATURE_DIR, "baseline_stats.npz")
OUTPUT_CSV = os.path.join(FEATURE_DIR, "val_drift_scores.csv")
OUTPUT_HIST = os.path.join(FEATURE_DIR, "val_drift_hist.png")
EPS = 1e-6
# =================================================

def mahalanobis_distance(x, mean, cov_inv):
    delta = x - mean
    return np.sqrt(delta.dot(cov_inv).dot(delta.T))

# ----------------- 加载 baseline -----------------
if not os.path.exists(BASELINE_STATS_FILE):
    raise FileNotFoundError(f"Baseline file not found: {BASELINE_STATS_FILE}")

baseline = np.load(BASELINE_STATS_FILE)
class_ids = sorted([k.split("_")[1] for k in baseline.keys() if k.endswith("_mean")])
print("Loaded baseline for classes:", class_ids)

# ----------------- 计算漂移 -----------------
all_scores = []
all_records = []

for cls_id in class_ids:
    # 特征文件
    feature_file = os.path.join(FEATURE_DIR, f"val_{cls_id}_features.npy")
    filename_file = os.path.join(FEATURE_DIR, f"val_{cls_id}_filenames.npy")

    if not os.path.exists(feature_file) or not os.path.exists(filename_file):
        print(f"⚠️ Val feature file not found for class {cls_id}: {feature_file}")
        continue

    X_val = np.load(feature_file)
    filenames = np.load(filename_file, allow_pickle=True)

    mean = baseline[f"class_{cls_id}_mean"]
    cov = baseline[f"class_{cls_id}_cov"]
    cov_inv = np.linalg.inv(cov)

    for i in range(len(X_val)):
        score = mahalanobis_distance(X_val[i], mean, cov_inv)
        all_scores.append(score)
        all_records.append({
            "filename": filenames[i],
            "class": cls_id,
            "drift_score": float(score)
        })

if len(all_scores) == 0:
    print("⚠️ No valid drift scores to summarize.")
else:
    all_scores = np.array(all_scores)

    # ----------------- 保存 CSV -----------------
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "drift_score"])
        writer.writeheader()
        writer.writerows(all_records)
    print("Saved drift scores:", OUTPUT_CSV)

    # ----------------- 可视化 -----------------
    plt.figure(figsize=(6,4))
    plt.hist(all_scores, bins=30, alpha=0.7)
    plt.xlabel("Drift Score")
    plt.ylabel("Count")
    plt.title("Val Drift Histogram")
    plt.tight_layout()
    plt.savefig(OUTPUT_HIST, dpi=150)
    print("Saved drift histogram:", OUTPUT_HIST)

    # ----------------- 打印统计 -----------------
    print("\n===== Drift Score Summary (VAL) =====")
    print("Total samples:", len(all_scores))
    print("Mean drift:  ", all_scores.mean())
    print("Std drift:   ", all_scores.std())
    print("Max drift:   ", all_scores.max())
    print("95% quantile:", np.quantile(all_scores, 0.95))
