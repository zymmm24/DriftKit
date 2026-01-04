import torch
import numpy as np
import pandas as pd
import pickle
import gc
from ultralytics import YOLO
from sklearn.decomposition import PCA  # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler  # pyright: ignore[reportMissingImports]
from pathlib import Path


class YOLO11AutoCollector:
    def __init__(self, model_path, dataset_root="dataset"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"📡 运行设备: {self.device}")

        self.model = YOLO(model_path)
        self.dataset_root = Path(dataset_root)
        self.label_map = self.model.names

        # 锁定分类头前一层
        self.target_layer_idx = self._lock_feature_layer()
        self._current_batch_features = []
        self._hook_handle = None
        self._register_hook()

    def _lock_feature_layer(self):
        layers = list(self.model.model.model)
        idx = len(layers) - 2
        print(f"🎯 锁定特征层: 索引 [{idx}], 类型 [{layers[idx].__class__.__name__}]")
        return idx

    def _hook_fn(self, module, input, output):
        if isinstance(output, (list, tuple)):
            output = output[0]
        feat = output.detach().cpu()
        if feat.dim() == 4:
            feat = torch.mean(feat, dim=[2, 3])
        self._current_batch_features.extend(feat.numpy())

    def _register_hook(self):
        layer = list(self.model.model.model)[self.target_layer_idx]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)

    def run(self):
        if not self.dataset_root.exists():
            print(f"❌ 路径不存在: {self.dataset_root}")
            return None

        img_list = [
            str(p) for p in self.dataset_root.rglob("*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        if not img_list:
            print(f"⚠️ 未找到图片。")
            return None

        print(f"🚀 开始处理 {len(img_list)} 张图片...")
        all_records = []

        results = self.model.predict(
            source=img_list,
            batch=1,
            imgsz=224,
            stream=True,
            device=self.device,
            verbose=False
        )

        for res in results:
            if not self._current_batch_features:
                print(f"⚠️ 未抓取到特征: {res.path}")
                continue

            img_emb = self._current_batch_features.pop(0)
            record = {
                "img_name": Path(res.path).name,
                "image_embedding": img_emb.astype(np.float32),
                "label": "unknown",
                "conf": 0.0
            }
            if hasattr(res, 'probs') and res.probs is not None:
                cls_id = int(res.probs.top1)
                record["label"] = self.label_map.get(cls_id, f"class_{cls_id}")
                record["conf"] = float(res.probs.top1conf)
            all_records.append(record)

            if len(all_records) % 100 == 0:
                print(f"已处理: {len(all_records)}")
                gc.collect()

        if self._hook_handle:
            self._hook_handle.remove()

        df = pd.DataFrame(all_records)
        print(f"✅ 特征提取完成: {df.shape}")
        return df

    def save_assets(self, df, folder="baseline_assets", pca=None, scaler=None):
        Path(folder).mkdir(parents=True, exist_ok=True)

        X = np.stack(df['image_embedding'].values)
        if pca is None or scaler is None:
            scaler = StandardScaler()
            n_comp = min(128, X.shape[0], X.shape[1])
            pca = PCA(n_components=n_comp)
            X_pca = pca.fit_transform(scaler.fit_transform(X))
            df['embedding_pca'] = list(X_pca.astype(np.float16))
            df.drop(columns=['image_embedding']).to_pickle(f"{folder}/baseline_db.pkl")
            with open(f"{folder}/pca_scaler.pkl", "wb") as f:
                pickle.dump({"scaler": scaler, "pca": pca, "names": self.label_map}, f)
            print(f"📦 训练集基准资产已保存至: {folder}")
        else:
            # 验证集使用已有 PCA + Scaler
            X_pca = pca.transform(scaler.transform(X))
            df['embedding_pca'] = list(X_pca.astype(np.float16))
            df.drop(columns=['image_embedding']).to_pickle(f"{folder}/val_test_data.pkl")
            print(f"📦 验证集 embedding 已保存至: {folder}/val_test_data.pkl")
        return pca, scaler


if __name__ == "__main__":
    MODEL_P = "runs/classify/train2/weights/best.pt"
    TRAIN_D = "dataset/train"
    VAL_D = "dataset/val"
    ASSET_DIR = "baseline_assets"

    # ---------- Step 1: 训练集 ----------
    print("\n[STEP 1] 生成训练集基准")
    coll_train = YOLO11AutoCollector(MODEL_P, TRAIN_D)
    df_train = coll_train.run()
    if df_train is not None:
        pca, scaler = coll_train.save_assets(df_train, folder=ASSET_DIR)

    # ---------- Step 2: 验证集 ----------
    print("\n[STEP 2] 生成验证集 embedding")
    coll_val = YOLO11AutoCollector(MODEL_P, VAL_D)
    df_val = coll_val.run()
    if df_val is not None:
        coll_val.save_assets(df_val, folder=ASSET_DIR, pca=pca, scaler=scaler)
