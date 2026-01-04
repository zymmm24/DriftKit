import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import pairwise_distances  # pyright: ignore[reportMissingImports]

# -----------------------------
# 工具函数
# -----------------------------
def compute_mmd(X, Y, gamma=1.0):
    """RBF Kernel MMD - 修正版本"""
    from sklearn.metrics.pairwise import rbf_kernel
    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)
    m = X.shape[0]; n = Y.shape[0]
    mmd = K_XX.sum()/(m*m) + K_YY.sum()/(n*n) - 2*K_XY.sum()/(m*n)
    return float(np.sqrt(max(mmd, 0.0)))

def feature_level_tests(baseline_emb, current_emb, alpha=0.05):
    changed_dims = []
    details = []
    for dim in range(baseline_emb.shape[1]):
        stat, p = ks_2samp(baseline_emb[:, dim], current_emb[:, dim])
        mean_diff = baseline_emb[:, dim].mean() - current_emb[:, dim].mean()
        pooled_std = np.sqrt((baseline_emb[:, dim].var() + current_emb[:, dim].var())/2)
        d = mean_diff / pooled_std if pooled_std>0 else 0
        details.append({
            "feature": f"dim_{dim}",
            "pval": float(p),
            "cohen_d": float(d)
        })
        if p < alpha and abs(d) > 0.3:
            changed_dims.append(f"dim_{dim}")
    return changed_dims, details

def nearest_neighbor_anomaly(baseline_emb, current_emb, current_names, top_k=50):
    dists = pairwise_distances(current_emb, baseline_emb)
    nn_dist = dists.min(axis=1)
    top_idx = np.argsort(nn_dist)[-top_k:][::-1]
    top_samples = [{"img_name": str(current_names[i]), "nn_dist": float(nn_dist[i])} for i in top_idx]
    return top_samples

# -----------------------------
# DriftReportGenerator
# -----------------------------
class DriftReportGenerator:
    def __init__(self, baseline_pkl, test_pkl):
        if not os.path.exists(baseline_pkl):
            raise FileNotFoundError(f"未找到基准数据: {baseline_pkl}")
        if not os.path.exists(test_pkl):
            raise FileNotFoundError(f"未找到测试数据: {test_pkl}")

        self.baseline_df = pd.read_pickle(baseline_pkl)
        self.test_df = pd.read_pickle(test_pkl)
        self.baseline_emb = np.stack(self.baseline_df["embedding_pca"].values)
        self.test_emb = np.stack(self.test_df["embedding_pca"].values)
        self.test_names = self.test_df["img_name"].values

    def generate_report(self, output_path="drift_report.json", alpha=0.05):
        try:
            # 全局 MMD 漂移
            mmd_score = compute_mmd(self.baseline_emb, self.test_emb)
            is_drift_global = bool(mmd_score > 0.01)
            status_global = "DRIFT DETECTED" if is_drift_global else "DATA STABLE"

            # 按类漂移
            per_class = {}
            classes = self.baseline_df["label"].unique()
            for cls in classes:
                try:
                    base_cls_emb = np.stack(self.baseline_df[self.baseline_df["label"]==cls]["embedding_pca"].values)
                    test_cls_emb = np.stack(self.test_df[self.test_df["label"]==cls]["embedding_pca"].values)
                    if len(test_cls_emb) > 0:
                        mmd_cls = compute_mmd(base_cls_emb, test_cls_emb)
                        per_class[int(cls)] = {
                            "baseline_size": int(len(base_cls_emb)),
                            "test_size": int(len(test_cls_emb)),
                            "mmd": float(mmd_cls),
                            "is_drift": bool(mmd_cls > 0.01)
                        }
                except Exception as e:
                    print(f"Warning: Failed to process class {cls}: {e}")
                    continue

            # 特征维度漂移
            try:
                changed_dims, feature_details = feature_level_tests(self.baseline_emb, self.test_emb, alpha=alpha)
            except Exception as e:
                print(f"Warning: Feature level tests failed: {e}")
                changed_dims, feature_details = [], []

            # 样本级漂移
            try:
                top_samples = nearest_neighbor_anomaly(self.baseline_emb, self.test_emb, self.test_names, top_k=50)
            except Exception as e:
                print(f"Warning: Nearest neighbor anomaly failed: {e}")
                top_samples = []

            # 生成报告
            report = {
                "meta": {
                    "generated_at": datetime.now().isoformat(),
                    "report_type": "YOLO Feature Drift Report",
                    "version": "v1.0"
                },
                "data_info": {
                    "baseline_source": "baseline_assets/baseline_db.pkl",
                    "test_source": "baseline_assets/val_test_data.pkl",
                    "baseline_size": int(len(self.baseline_emb)),
                    "test_size": int(len(self.test_emb))
                },
                "statistics": {
                    "mmd_score": float(mmd_score),
                    "alpha": float(alpha)
                },
                "decision": {
                    "is_drift": is_drift_global,
                    "status": status_global
                },
                "interpretation": (
                    "检测到漂移" if is_drift_global else
                    "当前数据分布与训练阶段保持一致，未发现显著特征漂移，模型运行状态稳定。"
                ),
                "per_class_drift": per_class,
                "feature_level_drift": {
                    "changed_dims": changed_dims,
                    "details": feature_details
                },
                "sample_level_drift": top_samples
            }

            # 保存 JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"📄 漂移报告已生成: {output_path}")
            return report

        except Exception as e:
            print(f"❌ 报告生成失败: {e}")
            import traceback
            traceback.print_exc()
            raise

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    try:
        baseline_path = "baseline_assets/baseline_db.pkl"
        test_path = "baseline_assets/val_test_data.pkl"
        generator = DriftReportGenerator(baseline_path, test_path)
        generator.generate_report()
        print("✅ 漂移报告生成成功！")
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
