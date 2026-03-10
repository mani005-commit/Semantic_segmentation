
# final training pipeline for VOC segmentation

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCDataset
from model import UNet
from loss import CombinedLoss
from metrics import dice_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset paths
image_folder = "datasets/VOC2012_train_val/VOC2012_train_val/JPEGImages"
mask_folder = "datasets/VOC2012_train_val/VOC2012_train_val/SegmentationClass"
split_file = "datasets/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/trainval.txt"


# load image ids
with open(split_file,"r") as f:
    image_ids = f.read().splitlines()


# train validation split
train_ids, val_ids = train_test_split(
    image_ids,
    test_size=0.2,
    random_state=42
)


# augmentations
train_transform = A.Compose([
    A.Resize(300,300),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.ImageCompression(quality_range=(60,100),p=0.3),
    A.Normalize(),
    ToTensorV2()
])


val_transform = A.Compose([
    A.Resize(300,300),
    A.Normalize(),
    ToTensorV2()
])


# datasets
train_dataset = VOCDataset(
    train_ids,
    image_folder,
    mask_folder,
    transform=train_transform
)

val_dataset = VOCDataset(
    val_ids,
    image_folder,
    mask_folder,
    transform=val_transform
)


# dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)


# model
model = UNet(num_classes=21).to(device)


# loss
criterion = CombinedLoss()


# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)


# learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=80
)


# mixed precision
scaler = torch.cuda.amp.GradScaler()


epochs = 80
best_dice = 0


for epoch in range(epochs):

    model.train()
    train_loss = 0


    for images, masks in tqdm(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(images)

            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()


    train_loss /= len(train_loader)


    model.eval()

    val_dice = 0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice = dice_score(outputs, masks)

            val_dice += dice


    val_dice /= len(val_loader)


    scheduler.step()


    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")


    if val_dice > best_dice:

        best_dice = val_dice

        os.makedirs("outputs/checkpoints",exist_ok=True)

        torch.save(
            model.state_dict(),
            "outputs/checkpoints/best_model.pth"
        )

        print("Best model saved")