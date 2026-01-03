# drift_detector.py
import os
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from scipy.stats import ks_2samp
from datetime import datetime

# -----------------------------
# 工具函数
# -----------------------------
def compute_mmd(X, Y, gamma=1.0):
    """RBF Kernel MMD"""
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    mmd_sq = np.mean(K_XX) + np.mean(K_YY) - 2*np.mean(K_XY)
    return np.sqrt(max(mmd_sq, 0.0))

def feature_level_tests(baseline_emb, current_emb, alpha=0.05):
    """KS-test + Cohen's d per feature dimension"""
    changed_dims, pvals, cohen_d = [], [], []
    for dim in range(baseline_emb.shape[1]):
        stat, p = ks_2samp(baseline_emb[:, dim], current_emb[:, dim])
        mean_diff = baseline_emb[:, dim].mean() - current_emb[:, dim].mean()
        pooled_std = np.sqrt((baseline_emb[:, dim].var() + current_emb[:, dim].var())/2)
        d = mean_diff / pooled_std if pooled_std > 0 else 0
        pvals.append(p)
        cohen_d.append(d)
        if p < alpha and abs(d) > 0.3:
            changed_dims.append(dim)
    return {'changed_dims': changed_dims, 'pvals': pvals, 'cohen_d': cohen_d}

def nearest_neighbor_anomaly(baseline_emb, current_emb, top_k=50):
    """Compute nearest neighbor distances from current to baseline"""
    dists = pairwise_distances(current_emb, baseline_emb)
    nn_dist = dists.min(axis=1)
    top_idx = np.argsort(nn_dist)[-top_k:][::-1]
    top_samples = [i for i in top_idx]
    return top_samples, nn_dist[top_idx]

# -----------------------------
# 核心检测器
# -----------------------------
class DriftDetector:
    def __init__(self, baseline_path="../baseline_assets/baseline_db.pkl"):
        self.baseline_df = pd.read_pickle(baseline_path)
        self.baseline_X = np.stack(self.baseline_df["embedding_pca"].values).astype(np.float32)
        self.baseline_labels = self.baseline_df['label'].values
        print(f"✅ 基准库加载成功: {len(self.baseline_X)} 样本")

    # -------- 全局漂移 MMD --------
    def _estimate_gamma(self, X, Y):
        combined = np.vstack([X, Y])
        dists = pdist(combined, metric="sqeuclidean")
        median_dist = np.median(dists)
        return 1.0 / median_dist if median_dist > 0 else 1.0

    def calculate_mmd(self, X, Y, gamma):
        return compute_mmd(X, Y, gamma)

    def run_permutation_test(self, X, Y, iterations=100):
        gamma = self._estimate_gamma(X, Y)
        observed_mmd = self.calculate_mmd(X, Y, gamma)
        combined = np.vstack([X, Y])
        n = X.shape[0]
        count = 0
        print(f"执行排列检验 ({iterations} 次迭代)...")
        for _ in range(iterations):
            idx = np.random.permutation(len(combined))
            new_X = combined[idx[:n]]
            new_Y = combined[idx[n:]]
            if self.calculate_mmd(new_X, new_Y, gamma) >= observed_mmd:
                count += 1
        p_value = count / iterations
        return observed_mmd, p_value

    # -------- 主检测流程 --------
    def detect(self, test_pkl_path, window_size=100, alpha=0.05, save_path="../baseline_assets/drift_result.pkl"):
        test_df = pd.read_pickle(test_pkl_path)
        test_X = np.stack(test_df['embedding_pca'].values).astype(np.float32)
        test_labels = test_df['label'].values

        # 随机窗口采样
        size = min(len(self.baseline_X), len(test_X), window_size)
        X_sub = self.baseline_X[np.random.choice(len(self.baseline_X), size, replace=False)]
        Y_sub = test_X[np.random.choice(len(test_X), size, replace=False)]

        # 全局 MMD
        mmd_score, p_val = self.run_permutation_test(X_sub, Y_sub)
        is_drift = p_val < alpha
        status = "DRIFT DETECTED" if is_drift else "DATA STABLE"
        print(f"\n[LOG] 全局 MMD={mmd_score:.4f}, p-value={p_val:.4f} -> {status}")

        # -------- 按类别漂移 --------
        per_class = {}
        classes = np.unique(self.baseline_labels)
        for cls in classes:
            base_cls_X = self.baseline_X[self.baseline_labels == cls]
            cur_cls_X = test_X[test_labels == cls]
            if len(cur_cls_X) < 5:
                continue
            mmd_cls = compute_mmd(base_cls_X, cur_cls_X)
            per_class[cls] = "DRIFT" if mmd_cls > 0.01 else "STABLE"

        # -------- 特征维度漂移 --------
        feature_level = feature_level_tests(self.baseline_X, test_X, alpha=alpha)
        print(f"[LOG] 特征维度漂移: {feature_level['changed_dims']}")

        # -------- 样本级漂移 --------
        top_samples, nn_dists = nearest_neighbor_anomaly(self.baseline_X, test_X)
        print(f"[LOG] 样本级漂移 top samples: {top_samples}")

        # -------- 保存中间结果 --------
        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline_size": len(self.baseline_X),
            "test_size": len(test_X),
            "window_size": size,
            "mmd_score": float(mmd_score),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "is_drift": bool(is_drift),
            "status": status,
            "baseline_source": test_pkl_path,
            "test_source": test_pkl_path,
            "per_class": per_class,
            "feature_level": feature_level,
            "sample_level": {"top_samples": top_samples, "nn_dists": nn_dists.tolist()},
        }

        with open(save_path, "wb") as f:
            pickle.dump(result, f)

        print(f"📝 中间漂移检测结果已保存至: {save_path}")
        return result

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    detector = DriftDetector()
    detector.detect("../baseline_assets/val_test_data.pkl")
