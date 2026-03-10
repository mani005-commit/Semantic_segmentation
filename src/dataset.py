
# Pascal VOC segmentation dataset loader

import os
import cv2
import torch
from torch.utils.data import Dataset


class VOCDataset(Dataset):

    def __init__(self, image_ids, image_dir, mask_dir, transform=None):
        """
        image_ids : list of image filenames without extension
        image_dir : directory containing JPEGImages
        mask_dir  : directory containing SegmentationClass
        transform : albumentations transform pipeline
        """

        self.image_ids = image_ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform


    def __len__(self):
        return len(self.image_ids)


    def __getitem__(self, idx):

        image_id = self.image_ids[idx]

        image_path = os.path.join(self.image_dir, image_id + ".jpg")
        mask_path = os.path.join(self.mask_dir, image_id + ".png")

        # load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load mask (grayscale)
        mask = cv2.imread(mask_path, 0)

        if image is None:
            raise ValueError(f"Image not found: {image_path}")

        if mask is None:
            raise ValueError(f"Mask not found: {mask_path}")

        # apply augmentations
        if self.transform is not None:

            augmented = self.transform(image=image, mask=mask)

            image = augmented["image"]
            mask = augmented["mask"]

        # convert mask to long tensor for CrossEntropyLoss
        mask = mask.long()

        # safety check (VOC has 21 classes)
        if mask.max() > 20:
            mask[mask > 20] = 0

        return image, mask