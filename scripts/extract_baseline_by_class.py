import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ===================== 配置区 =====================
DATASET_ROOT = r"D:/github/DriftKit/dataset"   # ⚠️ 用 raw string
MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 640
DEVICE = "cpu"   # 你目前是 CPU，后面可改 cuda
OUTPUT_DIR = "outputs"
HOOK_LAYER_INDEX = 10   # C2PSA（高语义 backbone 末端）
# ==================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- 加载模型 ----------------
model = YOLO(MODEL_PATH)
net = model.model
net.to(DEVICE)
net.eval()

# ---------------- Hook 容器 ----------------
feature_buffer = []

def hook_fn(module, inp, out):
    """
    out: [B, C, H, W]
    -> GAP -> [B, C]
    """
    with torch.no_grad():
        feat = out.mean(dim=[2, 3])  # Global Average Pooling
        feature_buffer.append(feat.cpu())

# 注册 hook
hook_handle = net.model[HOOK_LAYER_INDEX].register_forward_hook(hook_fn)

# ---------------- 图像预处理 ----------------
def load_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(DEVICE)

# ---------------- 主逻辑 ----------------
all_features = {}
all_filenames = {}  # 新增：保存真实文件名
total_images = 0

for split in ["train", "val"]:
    split_dir = os.path.join(DATASET_ROOT, split)
    print(f"\n== SCANNING {split_dir} ==")

    if not os.path.isdir(split_dir):
        print(f"❌ Not found: {split_dir}")
        continue

    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        print(f"  -> Class {cls}")
        cls_features = []
        cls_filenames = []  # 新增

        img_files = sorted(os.listdir(cls_dir))
        for f in tqdm(img_files, desc=f"{split}/{cls}", leave=False):
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            img_path = os.path.join(cls_dir, f)
            img = load_image(img_path)
            if img is None:
                continue

            feature_buffer.clear()
            with torch.no_grad():
                _ = net(img)

            if len(feature_buffer) == 0:
                print(f"⚠️ No feature captured for {img_path}")
                continue

            feat = feature_buffer[0].squeeze(0)  # [256]
            cls_features.append(feat.numpy())
            cls_filenames.append(f)  # 保存真实文件名
            total_images += 1

        if len(cls_features) > 0:
            cls_features = np.stack(cls_features).astype(np.float32)
            all_features[f"{split}_{cls}"] = cls_features
            all_filenames[f"{split}_{cls}"] = cls_filenames  # 保存对应文件名
            print(f"    ✔ Collected {cls_features.shape[0]} features")

print(f"\n✅ TOTAL IMAGES PROCESSED: {total_images}")

# ---------------- 保存特征 ----------------
for k, v in all_features.items():
    out_path = os.path.join(OUTPUT_DIR, f"{k}_features.npy")
    np.save(out_path, v)
    print(f"Saved: {out_path}, shape={v.shape}")

# ---------------- 保存对应文件名 ----------------
for k, v in all_filenames.items():
    out_path = os.path.join(OUTPUT_DIR, f"{k}_filenames.npy")
    np.save(out_path, np.array(v))  # 保存字符串数组
    print(f"Saved filenames: {out_path}, length={len(v)}")

# ---------------- 清理 ----------------
hook_handle.remove()

print("\n🎯 Baseline feature extraction DONE.")
