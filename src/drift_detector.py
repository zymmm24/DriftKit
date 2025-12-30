import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
# 使用非交互式后端，避免 PyCharm / interagg 崩溃
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import rbf_kernel
from scipy.spatial.distance import pdist


class YOLO11DriftDetector:
    def __init__(self, assets_path="baseline_assets"):
        self.assets_path = assets_path

        # 1. 加载基准库 (Train)
        print("正在加载基准资产...")
        self.baseline_df = pd.read_pickle(
            os.path.join(assets_path, "baseline_db.pkl")
        )
        self.baseline_X = np.stack(
            self.baseline_df["embedding_pca"].values
        ).astype(np.float32)

        # 2. 加载 PCA + Scaler
        with open(os.path.join(assets_path, "pca_scaler.pkl"), "rb") as f:
            assets = pickle.load(f)
            self.scaler = assets["scaler"]
            self.pca = assets["pca"]

        print(f"✅ 基准库加载成功: {len(self.baseline_X)} 样本")

    # -----------------------------
    # 核函数参数估计
    # -----------------------------
    def _estimate_gamma(self, X, Y):
        """
        Median Heuristic：自适应估计 RBF gamma
        """
        combined = np.vstack([X, Y])
        dists = pdist(combined, metric="sqeuclidean")
        median_dist = np.median(dists)
        return 1.0 / median_dist if median_dist > 0 else 1.0

    # -----------------------------
    # MMD 计算
    # -----------------------------
    def calculate_mmd(self, X, Y, gamma):
        """
        计算 MMD（返回 sqrt(MMD^2)）
        """
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)

        mmd_sq = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return np.sqrt(max(mmd_sq, 0.0))

    # -----------------------------
    # 排列检验
    # -----------------------------
    def run_permutation_test(self, X, Y, iterations=100):
        """
        Permutation Test 计算 P-Value
        """
        gamma = self._estimate_gamma(X, Y)
        observed_mmd = self.calculate_mmd(X, Y, gamma)

        combined = np.vstack([X, Y])
        n = X.shape[0]
        count = 0

        print(f"正在执行排列检验 ({iterations} 次迭代)...")
        for _ in range(iterations):
            idx = np.random.permutation(len(combined))
            new_X = combined[idx[:n]]
            new_Y = combined[idx[n:]]

            if self.calculate_mmd(new_X, new_Y, gamma) >= observed_mmd:
                count += 1

        p_value = count / iterations
        return observed_mmd, p_value

    # -----------------------------
    # 主检测流程
    # -----------------------------
    def detect(
        self,
        test_pkl_path,
        window_size=100,
        alpha=0.05,
        save_plot_path="drift_visualization.png",
        save_result_path="drift_result.pkl",
    ):
        """
        执行漂移检测并保存结果资产
        """
        test_df = pd.read_pickle(test_pkl_path)
        test_X = np.stack(
            test_df["embedding_pca"].values
        ).astype(np.float32)

        # 随机窗口采样（保证公平对比）
        size = min(len(self.baseline_X), len(test_X), window_size)

        X_sub = self.baseline_X[
            np.random.choice(len(self.baseline_X), size, replace=False)
        ]
        Y_sub = test_X[
            np.random.choice(len(test_X), size, replace=False)
        ]

        # MMD + 显著性检验
        mmd_score, p_val = self.run_permutation_test(X_sub, Y_sub)

        is_drift = p_val < alpha
        status = "DRIFT DETECTED" if is_drift else "DATA STABLE"

        print("\n" + "=" * 40)
        print(f"结果判决: {status}")
        print(f"MMD 距离: {mmd_score:.4f}")
        print(f"P-Value : {p_val:.4f} (alpha={alpha})")
        print("=" * 40)

        # 可视化
        self._plot_results(
            X_sub,
            Y_sub,
            mmd_score,
            p_val,
            status,
            save_plot_path,
        )

        # -----------------------------
        # 保存漂移检测结果（供报告使用）
        # -----------------------------
        result = {
            "timestamp": datetime.now().isoformat(),
            "baseline_size": int(len(self.baseline_X)),
            "test_size": int(len(test_X)),
            "window_size": int(size),
            "mmd_score": float(mmd_score),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "is_drift": bool(is_drift),
            "status": status,
            "baseline_source": os.path.join(
                self.assets_path, "baseline_db.pkl"
            ),
            "test_source": test_pkl_path,
            "visualization": save_plot_path,
        }

        with open(save_result_path, "wb") as f:
            pickle.dump(result, f)

        print(f"📝 漂移检测结果已保存至: {save_result_path}")

        return is_drift, mmd_score, p_val

    # -----------------------------
    # 可视化（保存，不 show）
    # -----------------------------
    def _plot_results(
        self,
        X,
        Y,
        score,
        p_val,
        status,
        save_path,
    ):
        plt.figure(figsize=(10, 6))

        plt.scatter(
            X[:, 0],
            X[:, 1],
            alpha=0.6,
            s=30,
            label="Baseline (Train)",
        )
        plt.scatter(
            Y[:, 0],
            Y[:, 1],
            alpha=0.6,
            s=30,
            label="Current (Test)",
        )

        title_color = "red" if status == "DRIFT DETECTED" else "green"
        plt.title(
            f"Drift Analysis\nMMD={score:.4f}, P-Value={p_val:.4f}\n{status}",
            color=title_color,
        )

        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.5)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"📊 漂移可视化已保存至: {save_path}")


if __name__ == "__main__":
    detector = YOLO11DriftDetector()
    detector.detect(
        test_pkl_path="../baseline_assets/val_test_data.pkl",
        window_size=100,
    )
