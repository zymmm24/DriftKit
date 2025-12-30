import os
import pickle
import json
from datetime import datetime


class DriftReportGenerator:
    def __init__(self, result_path="drift_result.pkl"):
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"未找到漂移检测结果文件: {result_path}"
            )

        with open(result_path, "rb") as f:
            self.result = pickle.load(f)

        print("✅ 已加载漂移检测结果")

    def generate(self, output_path="drift_report.json"):
        """
        生成结构化、可读的漂移报告
        """
        report = {
            "meta": self._build_meta(),
            "data_info": self._build_data_info(),
            "statistics": self._build_statistics(),
            "decision": self._build_decision(),
            "interpretation": self._build_interpretation(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📄 漂移报告已生成: {output_path}")
        return report

    # -----------------------------
    # 报告各组成部分
    # -----------------------------
    def _build_meta(self):
        return {
            "generated_at": datetime.now().isoformat(),
            "report_type": "YOLO Feature Drift Report",
            "version": "v1.0",
        }

    def _build_data_info(self):
        return {
            "baseline_source": self.result.get("baseline_source"),
            "test_source": self.result.get("test_source"),
            "baseline_size": self.result.get("baseline_size"),
            "test_size": self.result.get("test_size"),
            "window_size": self.result.get("window_size"),
        }

    def _build_statistics(self):
        return {
            "mmd_score": round(self.result["mmd_score"], 5),
            "p_value": round(self.result["p_value"], 5),
            "alpha": self.result["alpha"],
            "visualization": self.result.get("visualization"),
        }

    def _build_decision(self):
        return {
            "is_drift": self.result["is_drift"],
            "status": self.result["status"],
        }

    def _build_interpretation(self):
        """
        给“非算法人员”看的解释
        """
        if self.result["is_drift"]:
            return (
                "检测到当前数据分布与训练阶段存在显著差异。"
                "建议进一步定位漂移来源（类别、场景或特征维度），"
                "并评估是否需要重新训练或自适应调整模型。"
            )
        else:
            return (
                "当前数据分布与训练阶段保持一致，"
                "未发现显著特征漂移，模型运行状态稳定。"
            )


if __name__ == "__main__":
    generator = DriftReportGenerator("../drift_result.pkl")
    generator.generate("drift_report.json")
