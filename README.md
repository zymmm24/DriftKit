# DriftKit

**YOLO 感知特征分布漂移检测与解释框架**

DriftKit 是一个面向 YOLO 系列模型的特征分布漂移检测工具。在机器学习系统中，模型性能退化往往是"静默"发生的——输入数据的分布悄然偏离训练数据，而模型仍在正常输出预测结果，直到错误积累到无法忽视的程度。

DriftKit 通过提取模型内部特征表示，建立统计基准，实时监控新数据与基准的分布差异，在问题恶化之前发出预警。

---

## 适用场景

- **工业质检**：产品外观/缺陷检测模型的持续监控
- **安防监控**：摄像头场景变化（光照、角度、遮挡）的自动感知
- **自动驾驶**：环境数据漂移检测（天气、地域、季节变化）
- **医疗影像**：设备更换或参数调整后的数据一致性验证

---

## 核心功能

| 功能模块 | 描述 |
|----------|------|
| 自动特征挂载 | 自动定位 YOLO 分类头前层，无需手动配置即可提取高维特征 |
| 多层级漂移检测 | 支持全局、类别、特征维度、样本级四种粒度的漂移分析 |
| 统计显著性检验 | 基于 MMD + 排列检验，区分随机波动与真实漂移 |
| 结构化报告输出 | 生成 JSON 格式报告，包含人类可读的决策建议 |
| 空间对齐机制 | 使用持久化的 Scaler + PCA 确保新旧数据可比性 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DriftKit Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   训练数据    │    │   YOLO 模型   │    │   验证/线上   │               │
│  │  (Baseline)  │    │  (Backbone)  │    │    数据      │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │              Step 1: Baseline Collector                     │         │
│  │  • Hook 特征层 → 提取 Embedding → Scaler + PCA 降维         │         │
│  └────────────────────────────┬───────────────────────────────┘         │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                               │
│                    │   baseline_assets/  │                               │
│                    │  ├─ baseline_db.pkl │                               │
│                    │  ├─ pca_scaler.pkl  │                               │
│                    │  └─ val_test_data   │                               │
│                    └──────────┬──────────┘                               │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │              Step 2: Drift Detector                         │         │
│  │  • MMD 分布差异 → 排列检验 → 多层级漂移分析                  │         │
│  └────────────────────────────┬───────────────────────────────┘         │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │              Step 3: Drift Report                           │         │
│  │  • 统计结果 → 决策判定 → JSON 报告 + 人类可读解释            │         │
│  └────────────────────────────────────────────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 安装

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，支持 CPU 运行)

### 安装依赖

```bash
git clone https://github.com/zymmm24/driftkit.git
cd DriftKit
pip install -r requirements.txt
```

### 依赖清单

```
matplotlib==3.10.8
numpy==2.4.0
pandas==2.3.3
scikit_learn==1.8.0
scipy==1.16.3
torch==2.6.0
ultralytics==8.2.38
```

---

## 快速开始

### 1. 准备数据集

按以下结构组织数据：

```
dataset/
├── train/          # 训练集（用于建立基准）
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/            # 验证集（用于检测漂移）
    ├── class_1/
    ├── class_2/
    └── ...
```

### 2. 训练 YOLO 分类模型（可选）

如果已有训练好的模型，可跳过此步骤。

```bash
yolo classify train data=./dataset model=yolo11n-cls.pt epochs=50 imgsz=224
```

### 3. 运行完整流程

```bash
cd src

# Step 1: 生成基准资产
python baseline_collector.py

# Step 2: 执行漂移检测
python drift_detector.py

# Step 3: 生成漂移报告
python drift_report.py
```

---

## 使用方法

### Step 1: 基准特征收集

`baseline_collector.py` 负责从训练数据中提取特征基准。

```python
from baseline_collector import YOLO11AutoCollector

collector = YOLO11AutoCollector(
    model_path="runs/classify/train2/weights/best.pt",
    dataset_root="dataset/train"
)

df = collector.run()
pca, scaler = collector.save_assets(df, folder="baseline_assets")
```

**输出文件：**

| 文件 | 描述 |
|------|------|
| `baseline_db.pkl` | 基准数据库，包含 label、embedding_pca、conf |
| `pca_scaler.pkl` | 空间映射器，包含 StandardScaler 和 PCA 参数 |

### Step 2: 漂移检测

`drift_detector.py` 执行分布差异检测。

```python
from drift_detector import DriftDetector

detector = DriftDetector(baseline_path="baseline_assets/baseline_db.pkl")

result = detector.detect(
    test_pkl_path="baseline_assets/val_test_data.pkl",
    window_size=100,
    alpha=0.05
)

if result["is_drift"]:
    print("检测到数据漂移！")
else:
    print("数据分布稳定")
```

**检测结果结构：**

```python
{
    "mmd_score": 0.0234,      # MMD 距离值
    "p_value": 0.03,          # 统计显著性
    "is_drift": True,         # 是否漂移
    "status": "DRIFT DETECTED",
    "per_class": {...},       # 按类别漂移结果
    "feature_level": {...},   # 特征维度漂移
    "sample_level": {...}     # 样本级异常
}
```

### Step 3: 报告生成

`drift_report.py` 生成结构化的 JSON 报告。

```python
from drift_report import DriftReportGenerator

generator = DriftReportGenerator(
    baseline_pkl="baseline_assets/baseline_db.pkl",
    test_pkl="baseline_assets/val_test_data.pkl"
)

report = generator.generate_report(output_path="drift_report.json")
```

---

## API 文档

### YOLO11AutoCollector

基准特征收集器类。

```python
class YOLO11AutoCollector:
    def __init__(self, model_path: str, dataset_root: str = "dataset")
    def run(self) -> pd.DataFrame
    def save_assets(self, df, folder="baseline_assets", pca=None, scaler=None)
```

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `__init__` | `model_path`, `dataset_root` | - | 初始化收集器，自动挂载特征层 Hook |
| `run` | - | `DataFrame` | 批量处理图片，返回特征数据 |
| `save_assets` | `df`, `folder`, `pca`, `scaler` | `(pca, scaler)` | 持久化基准资产 |

### DriftDetector

漂移检测器类。

```python
class DriftDetector:
    def __init__(self, baseline_path: str)
    def detect(self, test_pkl_path, window_size=100, alpha=0.05, save_path=...) -> dict
```

| 方法 | 参数 | 返回值 | 描述 |
|------|------|--------|------|
| `__init__` | `baseline_path` | - | 加载基准数据库 |
| `detect` | `test_pkl_path`, `window_size`, `alpha` | `dict` | 执行完整漂移检测流程 |

### DriftReportGenerator

报告生成器类。

```python
class DriftReportGenerator:
    def __init__(self, baseline_pkl: str, test_pkl: str)
    def generate_report(self, output_path: str, alpha=0.05) -> dict
```

---

## 报告结构示例

```json
{
  "meta": {
    "generated_at": "2025-12-31T20:06:13",
    "report_type": "YOLO Feature Drift Report",
    "version": "v1.0"
  },
  "data_info": {
    "baseline_size": 640,
    "test_size": 162
  },
  "statistics": {
    "mmd_score": 0.0077,
    "alpha": 0.05
  },
  "decision": {
    "is_drift": false,
    "status": "DATA STABLE"
  },
  "interpretation": "当前数据分布与训练阶段保持一致，未发现显著特征漂移，模型运行状态稳定。",
  "per_class_drift": {
    "1": { "mmd": 0.029, "is_drift": true },
    "2": { "mmd": 0.023, "is_drift": true }
  },
  "feature_level_drift": {
    "changed_dims": ["dim_106"]
  },
  "sample_level_drift": [
    { "img_name": "xxx.jpg", "nn_dist": 14.18 }
  ]
}
```

---

## 技术原理

### MMD (Maximum Mean Discrepancy)

MMD 是一种非参数的两样本检验方法，通过比较两个分布在再生核希尔伯特空间 (RKHS) 中的均值嵌入来判断它们是否来自同一分布。

```
MMD²(P, Q) = E[k(X, X')] + E[k(Y, Y')] - 2E[k(X, Y)]
```

其中 `k(·, ·)` 为 RBF 核函数。

### 排列检验 (Permutation Test)

通过随机打乱样本标签（来自基准/测试）计算零假设下的 MMD 分布，从而得到观测 MMD 的 p-value。

### 多层级漂移分析

| 层级 | 方法 | 用途 |
|------|------|------|
| 全局 | MMD + 排列检验 | 整体分布是否变化 |
| 类别 | 按类 MMD | 哪些类别发生漂移 |
| 特征 | KS 检验 + Cohen's d | 哪些特征维度异常 |
| 样本 | 最近邻距离 | 识别离群样本 |

---

## 项目结构

```
DriftKit/
├── src/                          # 核心源代码
│   ├── __init__.py
│   ├── baseline_collector.py     # 基准特征收集器
│   ├── drift_detector.py         # 漂移检测器
│   └── drift_report.py           # 报告生成器
├── baseline_assets/              # 持久化资产
│   ├── baseline_db.pkl           # 基准数据库
│   ├── pca_scaler.pkl            # 空间映射器
│   ├── val_test_data.pkl         # 验证集数据
│   └── drift_result.pkl          # 检测结果
├── dataset/                      # 数据集目录
│   ├── train/                    # 训练集
│   └── val/                      # 验证集
├── runs/                         # YOLO 训练输出
├── drift_report.json             # 漂移报告示例
├── requirements.txt              # 依赖清单
└── README.md                     # 项目文档
```

---

## 路线图

- [x] 基准特征自动收集
- [x] MMD 全局漂移检测
- [x] 排列检验统计显著性
- [x] 多层级漂移分析（类别/特征/样本）
- [x] JSON 结构化报告生成
- [ ] PCA 可视化模块
- [ ] 在线流式检测支持
- [ ] 自动告警与通知集成
- [ ] Web Dashboard
- [ ] 支持更多模型架构（RT-DETR、YOLO-World）

---

## 贡献

欢迎贡献代码、报告问题或提出改进建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 许可证

本项目采用 MIT 许可证。
