import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# ===================== 配置区 =====================
DATASET_ROOT = r"D:/github/DriftKit/dataset"
MODEL_PATH = "yolo11n.pt"
IMG_SIZE = 640
DEVICE = "cpu"          # 后面可切 cuda
HOOK_LAYER_INDEX = 10   # C2PSA（backbone 高语义层）

# 输出目录：明确区分
TRAIN_OUT_DIR = "outputs/train"
VAL_OUT_DIR   = "outputs/val"

os.makedirs(TRAIN_OUT_DIR, exist_ok=True)
os.makedirs(VAL_OUT_DIR, exist_ok=True)
# =================================================

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

# =================================================
# 核心函数：抽取一个 split 的特征
# =================================================
def extract_split_features(split_name, output_dir):
    """
    split_name: 'train' or 'val'
    output_dir: 对应输出目录
    """
    split_dir = os.path.join(DATASET_ROOT, split_name)
    print(f"\n== EXTRACTING {split_name.upper()} FEATURES ==")

    if not os.path.isdir(split_dir):
        print(f"❌ Not found: {split_dir}")
        return 0

    total_images = 0

    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        print(f"  -> Class {cls}")
        cls_features = []

        img_files = sorted(os.listdir(cls_dir))
        for f in tqdm(img_files, desc=f"{split_name}/{cls}", leave=False):
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
                print(f"⚠️ No feature captured: {img_path}")
                continue

            feat = feature_buffer[0].squeeze(0)  # [C]
            cls_features.append(feat.numpy())
            total_images += 1

        if len(cls_features) == 0:
            print(f"    ⚠️ No valid images for class {cls}")
            continue

        cls_features = np.stack(cls_features).astype(np.float32)
        out_path = os.path.join(output_dir, f"{cls}_features.npy")
        np.save(out_path, cls_features)

        print(f"    ✔ Saved {out_path}, shape={cls_features.shape}")

    return total_images

# =================================================
# 主流程（⚠️ 关键逻辑）
# =================================================

# Step 1 ✅ 只用 train 建立 baseline 特征
num_train = extract_split_features(
    split_name="train",
    output_dir=TRAIN_OUT_DIR
)

# Step 2 ✅ val 作为“新数据”，不参与 baseline
num_val = extract_split_features(
    split_name="val",
    output_dir=VAL_OUT_DIR
)

hook_handle.remove()

print("\n================ SUMMARY ================")
print(f"TRAIN images processed: {num_train}")
print(f"VAL   images processed: {num_val}")
print("🎯 Feature extraction finished (train = baseline, val = new data)")
