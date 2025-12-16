# 평가 의미 없음 테스트 데이터 안 나눠놔서 전체 이미지로 평가중..
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp


# ================================================================
# 1) 데이터셋 정의 (학습 때 사용한 transform 중 테스트용만 다시 정의)
# ================================================================
IMAGE_SIZE = 1024

class BeadTestDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        self.mask_dir = mask_dir

    def get_mask_path(self, img_path):
        base = os.path.splitext(os.path.basename(img_path))[0]
        for ext in [".png", ".jpg"]:
            cand = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"Mask not found for {img_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.get_mask_path(img_path)

        # 이미지 로드 + RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.float32) / 255.0

        # 정규화
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # HWC → CHW
        image = np.transpose(image, (2, 0, 1))

        # 마스크 로드 (0/255)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# ================================================================
# 2) IoU / Dice 계산 함수
# ================================================================
def compute_metrics(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    preds_f = preds.view(preds.size(0), -1)
    targets_f = targets.view(targets.size(0), -1)

    intersection = (preds_f * targets_f).sum(dim=1)
    union = preds_f.sum(dim=1) + targets_f.sum(dim=1) - intersection

    iou  = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (preds_f.sum(dim=1) + targets_f.sum(dim=1) + eps)

    return iou.mean().item(), dice.mean().item()


# ================================================================
# 3) 메인 평가 루틴
# ================================================================
def evaluate_model(model_path, image_dir, mask_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 모델 정의 (훈련과 완전히 동일해야 함)
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # 로드 후 weight 덮어씌워짐
        in_channels=3,
        classes=1,
    )
    model.to(device)

    # 가중치 로드
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 모델 로드 완료:", model_path)

    # 데이터셋 & 로더
    dataset = BeadTestDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_iou = 0.0
    total_dice = 0.0
    count = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device).float()   #  여기 .float() 추가
            masks  = masks.to(device).float()    # (마스크도 float로 맞춰주면 안전)

            logits = model(images)
            iou, dice = compute_metrics(logits, masks)

            total_iou  += iou
            total_dice += dice
            count += 1

    print("\n=========== 📊 최종 Test 성능 ===========")
    print(f"Mean IoU  : {total_iou / count:.4f}")
    print(f"Mean Dice : {total_dice / count:.4f}")
    print("=========================================\n")


# ================================================================
# 실행
# ================================================================
if __name__ == "__main__":
    evaluate_model(
        model_path="/home/ho/BEADtrain/modelresult/1127unet.pth",
        image_dir="/home/ho/BEADtrain/REAL/fitimage",
        mask_dir="/home/ho/BEADtrain/REAL/fitmask",
    )
