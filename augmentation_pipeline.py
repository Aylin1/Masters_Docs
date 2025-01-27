import numpy as np
import matplotlib.pyplot as plt
import os
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate,
    RandomBrightnessContrast, GaussNoise, Blur, MotionBlur
)

def augment_dataset(X, Y, config):
    """ 
    Apply augmentation to RGBI and CHM while ensuring alignment with masks.
    
    Args:
        X (np.ndarray): Input images (RGBI + CHM).
        Y (np.ndarray): Ground truth masks.
        config (dict): Augmentation settings.

    Returns:
        np.ndarray, np.ndarray: Augmented images and masks.
    """
    X_aug, Y_aug = [], []

    for i in range(len(X)):
        rgbi = X[i][..., :4]  # Bands 1-4 (RGBI)
        chm = X[i][..., 4:]   # Band 5 (CHM)
        mask = Y[i]           # Ground truth mask

        # Define spatial augmentation pipeline (applies to both X & Y)
        spatial_augment = Compose([
            HorizontalFlip(p=config.get("HorizontalFlip", 0)),
            VerticalFlip(p=config.get("VerticalFlip", 0)),
            ShiftScaleRotate(
                shift_limit=config.get("Shift", {}).get("shift_limit", 0),
                scale_limit=config.get("Scale", {}).get("scale_limit", 0),
                rotate_limit=config.get("Rotate", {}).get("rotate_limit", 0),
                p=1.0
            )
        ], additional_targets={'mask': 'mask'})  # Ensures same transformation on Y

        # Define RGBI-only augmentations (brightness, contrast, noise, blur)
        rgbi_augment = Compose([
            RandomBrightnessContrast(
                brightness_limit=config.get("RandomBrightness", {}).get("brightness_limit", 0),
                contrast_limit=config.get("RandomContrast", {}).get("contrast_limit", 0),
                p=1.0
            ),
            GaussNoise(var_limit=config.get("GaussNoise", {}).get("var_limit", (0, 0)), p=1.0),
            Blur(blur_limit=config.get("Blur", {}).get("blur_limit", 0), p=1.0),
            MotionBlur(blur_limit=config.get("MotionBlur", {}).get("blur_limit", 0), p=1.0)
        ])

        # Define CHM-specific augmentations (weaker noise, blur)
        chm_augment = Compose([
            GaussNoise(var_limit=(5.0, 20.0), p=config.get("GaussNoise", {}).get("p", 0)),
            Blur(blur_limit=3, p=config.get("Blur", {}).get("p", 0))
        ])

        # Apply spatial augmentations to RGBI, CHM, and Mask
        augmented = spatial_augment(image=rgbi, mask=mask)
        rgbi_aug = rgbi_augment(image=augmented["image"])["image"]
        chm_aug = chm_augment(image=chm)["image"]  # CHM gets separate augment

        # Merge augmented bands back together
        X_aug.append(np.concatenate([rgbi_aug, chm_aug], axis=-1))
        Y_aug.append(augmented["mask"])  # Ensures mask receives correct transformation

    return np.array(X_aug), np.array(Y_aug)
