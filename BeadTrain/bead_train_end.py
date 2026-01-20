#==============================================
# import
#==============================================
import os
import glob
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from datetime import datetime

#==============================================
# 0) CONFIG
#==============================================
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

MODEL_SAVE_ROOT = "/home/ho/BEADtrain/modelresult"
MODEL_SAVE_DIR  = os.path.join(MODEL_SAVE_ROOT, RUN_TIMESTAMP)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "unet_best.pth")
LOSS_FIG_PATH   = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")

DATA_ROOT = "/home/ho/BEADtrain/REAL"
IMAGE_DIR = f"{DATA_ROOT}/end/endimg"
MASK_DIR  = f"{DATA_ROOT}/end/endmask"

IMAGE_SIZE = 1280          # 모델 입력 크기 (정사각)
BATCH_SIZE = 2             # 1024면 1~2가 안정적 (원래 6은 터질 확률 높음)
NUM_EPOCHS = 60

# EarlyStopping
EARLY_STOP_PATIENCE = 7
EARLY_STOP_DELTA    = 1e-4

# Aug 옵션: end 라벨이 "방향 의미"가 있으면 flip은 끄는 게 안전
USE_HFLIP = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#==============================================
# 1) 데이터 경로 로드 / 매칭
#==============================================
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_DIR, "*.png"))
)
print("총 이미지 개수:", len(image_paths))

def get_mask_path(img_path):
    base = os.path.splitext(os.path.basename(img_path))[0]
    for ext in [".png", ".jpg"]:
        cand = os.path.join(MASK_DIR, base + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Mask not found for {img_path}")

# shuffle + split
random.shuffle(image_paths)
n_total = len(image_paths)

train_ratio = 0.7
val_ratio   = 0.15

n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val

train_imgs = image_paths[:n_train]
val_imgs   = image_paths[n_train:n_train+n_val]
test_imgs  = image_paths[n_train+n_val:]

print("Train:", len(train_imgs), "Val:", len(val_imgs), "Test:", len(test_imgs))

#==============================================
# 2) Transform (핵심 수정: 비율 유지 + 패딩)
#==============================================
def build_train_transform(image_size=1024, use_hflip=False):
    tfms = [
        # ✅ 비율 유지 리사이즈 + 패딩 (왜곡 방지)
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    ]

    if use_hflip:
        tfms.append(A.HorizontalFlip(p=0.5))

    tfms += [
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.05,
            rotate_limit=3,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8
        ),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=5, p=0.15),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    return A.Compose(tfms)

def build_val_transform(image_size=1024):
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),

        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

train_transform = build_train_transform(IMAGE_SIZE, use_hflip=USE_HFLIP)
val_transform   = build_val_transform(IMAGE_SIZE)

#==============================================
# 3) Dataset
#==============================================
class BeadDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = get_mask_path(img_path)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        # ✅ 0/1로 확실히 이진화 (회색값/안티앨리어싱 제거)
        mask = (mask > 127).astype(np.float32)   # 0 or 1
        mask = np.expand_dims(mask, axis=-1)     # (H,W,1)


        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]       # torch (C,H,W)
            mask  = augmented["mask"]        # torch (H,W,1) or (H,W)

        # mask -> (1,H,W)
        if isinstance(mask, np.ndarray):
            # 혹시 ToTensorV2가 없으면 대비
            if mask.ndim == 2:
                mask = mask[None, ...]
            else:
                mask = np.transpose(mask, (2, 0, 1))
            mask = torch.from_numpy(mask).float()
        else:
            # torch tensor
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.permute(2, 0, 1)

        return image, mask

train_dataset = BeadDataset(train_imgs, transform=train_transform)
val_dataset   = BeadDataset(val_imgs,   transform=val_transform)
test_dataset  = BeadDataset(test_imgs,  transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1,
                          shuffle=False, num_workers=2, pin_memory=True)

#==============================================
# 4) device / model
#==============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
).to(device)

#==============================================
# 5) Loss (Dice + BCE) + pos_weight 자동 계산
#==============================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (preds * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            preds.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1 - dice.mean()

def estimate_pos_weight(train_imgs, max_samples=200):
    # train 마스크 픽셀 비율로 pos_weight = neg/pos 근사
    sample_imgs = train_imgs[:min(len(train_imgs), max_samples)]
    pos = 0
    neg = 0
    for ip in sample_imgs:
        mp = get_mask_path(ip)
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        # 0/255 -> 0/1
        m = (m > 127).astype(np.uint8)
        pos += int(m.sum())
        neg += int(m.size - m.sum())

    if pos == 0:
        return 1.0  # 전부 배경이면 fallback
    return float(neg / pos)

pos_w_value = estimate_pos_weight(train_imgs)
pos_w_value = min(pos_w_value, 50.0)   # ✅ 상한 (너무 커지면 학습 불안정/부분 예측 가능)
pos_weight = torch.tensor([pos_w_value], device=device)
print(f"[pos_weight] estimated(clipped): {pos_w_value:.3f}")


bce_loss  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
dice_loss = DiceLoss()

def total_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)

#==============================================
# 6) Optim / Scheduler
#==============================================
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

#==============================================
# 7) Train / Val
#==============================================
def train_one_epoch(model, loader):
    model.train()
    epoch_loss = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = total_loss(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)

def validate(model, loader):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            loss = total_loss(logits, masks)
            epoch_loss += loss.item() * images.size(0)

    return epoch_loss / len(loader.dataset)

def compute_iou_dice_from_logits(logits, targets, threshold=0.4, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets_bin = (targets > 0.5).float()

    preds_flat   = preds.view(preds.size(0), -1)
    targets_flat = targets_bin.view(targets_bin.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1) - intersection

    iou  = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + eps)

    return iou.mean().item(), dice.mean().item()

#==============================================
# 8) Loop + EarlyStopping + Best Save
#==============================================
best_val_loss = np.inf
train_loss_history = []
val_loss_history   = []
no_improve_count = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader)
    val_loss   = validate(model, val_loader)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)

    print(f"[{epoch:02d}/{NUM_EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    scheduler.step(val_loss)

    if val_loss < (best_val_loss - EARLY_STOP_DELTA):
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("  ✅ Best model updated:", MODEL_SAVE_PATH)
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"  ⏳ No improvement: {no_improve_count}/{EARLY_STOP_PATIENCE}")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print("🛑 Early stopping triggered!")
            break

#==============================================
# 9) Loss Curve Save
#==============================================
plt.figure()
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history,   label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(LOSS_FIG_PATH, dpi=200)
plt.close()
print("📉 Loss curve saved:", LOSS_FIG_PATH)

#==============================================
# 10) Test Eval (Best model)
#==============================================
state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Best model loaded:", MODEL_SAVE_PATH)

total_iou, total_dice, n = 0.0, 0.0, 0
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        masks  = masks.to(device)

        logits = model(images)
        iou, dice = compute_iou_dice_from_logits(logits, masks)

        total_iou  += iou
        total_dice += dice
        n += 1

avg_iou  = total_iou  / max(n, 1)
avg_dice = total_dice / max(n, 1)

print("==============================")
print(f"📊 Test Mean IoU : {avg_iou:.4f}")
print(f"📊 Test Mean Dice: {avg_dice:.4f}")
print("==============================")
