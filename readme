# DriftKit

针对 YOLO 模型部署阶段的工具集合，已实现的功能包括在**参考基线阶段**抽取中间层特征、基线统计建模，以及基于基线对推理/测试期特征进行对比分析（漂移/距离度量）。

## 目前已实现的功能
- 在参考基线阶段运行 YOLO 模型并挂 hook 抽取中间层 embedding，导出每个检测实例的特征向量与元信息。
- 基于参考基线阶段的特征计算并保存基线统计（类条件均值/协方差或等效统计表示）。
- 对测试/推理期的特征与基线进行对比，计算漂移/距离度量并导出分析结果或可视化产物。
- 提供一个工具用于快速查看 `.npy` / `.npz` 特征文件的基本信息。

## 目录
```
DriftKit/
├── scripts/
│ ├── extract_baseline_by_class.py # 从数据集运行模型并抽取特征，输出 features.npy + meta.csv
│ ├── build_baseline_stats.py # 从抽取的特征构建基线统计并保存
│ └── analyze_features.py # 比较测试/推理期特征与基线，计算距离/漂移并导出结果
├── dataset/ # 示例数据集或数据说明
├── ultralytics-main/ 
├── inspect_npy.py # 快速查看 .npy/.npz 文件的 shape 与统计信息
├── yolo11n.pt / yolo11n-cls.pt # 示例/测试用的模型权重
└── README.md # 项目说明
```

## 输入 / 输出

- `scripts/extract_baseline_by_class.py`  
  - 输入：YOLO 权重、数据集路径、输出目录。  
  - 输出：`features.npy`（N × C 的特征矩阵）和 `meta.csv`（每行特征的元信息：image, class_id, bbox, conf_mean/conf_std, num_boxes 等）。

- `scripts/build_baseline_stats.py`  
  - 输入：`features.npy` 与 `meta.csv`。  
  - 输出：基线统计文件（例如 `baseline_stats.npz`），包含每类的均值、协方差或其它用于距离计算的数据结构。

- `scripts/analyze_features.py`  
  - 输入：基线统计文件、测试/推理期抽取的特征。  
  - 输出：每个测试样本/检测实例的漂移/距离评分文件及可视化结果。

- `inspect_npy.py`  
  - 输入：`.npy` / `.npz` 文件路径。  
  - 输出：终端打印的文件 shape、dtype、min/max/mean/std 等信息。

## 运行示例（按脚本名展示）
```bash
# 抽取参考基线阶段特征
python scripts/extract_baseline_by_class.py --weights <weights.pt> --dataset <train_dataset_path> --out_dir outputs/

# 构建基线统计
python scripts/build_baseline_stats.py --features outputs/features.npy --meta outputs/meta.csv --out_dir outputs/

# 分析 / 对比测试特征
python scripts/analyze_features.py --baseline outputs/baseline_stats.npz --test_features test_features.npy --out_dir analysis_results/

# 快速查看 .npy 文件
python inspect_npy.py outputs/features.npy
