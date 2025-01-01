import numpy as np
import matplotlib.pyplot as plt
from albumentations import Compose, RandomRotate90, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, GaussNoise, ElasticTransform, CoarseDropout
from albumentations.pytorch import ToTensorV2
from tensorflow.keras import models

def get_augmentation_pipeline(config):
    """
    Generate an augmentation pipeline based on a configuration.

    Args:
        config (dict): Dictionary specifying the augmentations and their parameters.

    Returns:
        Compose: An Albumentations augmentation pipeline.
    """
    augmentations = []

    if config.get("RandomRotate90", False):
        augmentations.append(RandomRotate90(p=config["RandomRotate90"]))

    if config.get("HorizontalFlip", False):
        augmentations.append(HorizontalFlip(p=config["HorizontalFlip"]))

    if config.get("VerticalFlip", False):
        augmentations.append(VerticalFlip(p=config["VerticalFlip"]))

    if config.get("ShiftScaleRotate", False):
        params = config["ShiftScaleRotate"]
        augmentations.append(ShiftScaleRotate(
            shift_limit=params.get("shift_limit", 0.01),
            scale_limit=params.get("scale_limit", 0.05),
            rotate_limit=params.get("rotate_limit", 15),
            p=params.get("p", 0.5)
        ))

    if config.get("RandomBrightnessContrast", False):
        params = config["RandomBrightnessContrast"]
        augmentations.append(RandomBrightnessContrast(
            brightness_limit=params.get("brightness_limit", 0.2),
            contrast_limit=params.get("contrast_limit", 0.2),
            p=params.get("p", 0.5)
        ))

    if config.get("GaussNoise", False):
        params = config["GaussNoise"]
        augmentations.append(GaussNoise(
            var_limit=params.get("var_limit", (10.0, 50.0)),
            p=params.get("p", 0.3)
        ))

    if config.get("ElasticTransform", False):
        params = config["ElasticTransform"]
        augmentations.append(ElasticTransform(
            alpha=params.get("alpha", 1),
            sigma=params.get("sigma", 50),
            alpha_affine=params.get("alpha_affine", 50),
            p=params.get("p", 0.2)
        ))

    if config.get("CoarseDropout", False):
        params = config["CoarseDropout"]
        augmentations.append(CoarseDropout(
            max_holes=params.get("max_holes", 8),
            max_height=params.get("max_height", 16),
            max_width=params.get("max_width", 16),
            p=params.get("p", 0.5)
        ))

    return Compose(augmentations)

def augment_dataset(X, Y, config, num_versions=2):
    """
    Augment the dataset based on the given configuration.

    Args:
        X (np.ndarray): Training images.
        Y (np.ndarray): Training masks.
        config (dict): Augmentation configuration.
        num_versions (int): Number of augmented versions per image.

    Returns:
        np.ndarray: Augmented training images.
        np.ndarray: Augmented training masks.
    """
    augmentation_pipeline = get_augmentation_pipeline(config)
    X_aug, Y_aug = [], []
    for i in range(len(X)):
        for _ in range(num_versions):
            augmented = augmentation_pipeline(image=X[i], mask=Y[i])
            X_aug.append(augmented['image'])
            Y_aug.append(augmented['mask'])
    return np.array(X_aug), np.array(Y_aug)

def normalize_image(image):
    """
    Normalize image pixel values for proper visualization.

    Args:
        image (np.ndarray): Image to normalize.

    Returns:
        np.ndarray: Normalized image.
    """
    image = np.clip(image, 0, 1)
    return image

def plot_original_and_augmented_versions(X, X_aug, num_samples=3, num_versions=3):
    """
    Plot original and augmented versions of images (first 3 channels only).

    Args:
        X (np.ndarray): Original images.
        X_aug (np.ndarray): Augmented images.
        num_samples (int): Number of original images to display.
        num_versions (int): Number of augmented versions per image.
    """
    fig, axs = plt.subplots(num_samples, num_versions + 1, figsize=(5 * (num_versions + 1), 5 * num_samples))

    for i in range(num_samples):
        axs[i, 0].imshow(normalize_image(X[i][..., :3]))
        axs[i, 0].set_title(f"Original Image {i+1}")
        axs[i, 0].axis('off')

        for j in range(num_versions):
            axs[i, j + 1].imshow(normalize_image(X_aug[i * num_versions + j][..., :3]), aspect='auto')
            axs[i, j + 1].set_title(f"Augmented Version {j+1}")
            axs[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()

def run_augmentation_and_train(X, Y, X_val, Y_val, augment_configs, build_model_fn, num_versions=10, batch_size=8, epochs=25):
    """
    Apply augmentations and train models based on the configurations.

    Args:
        X (np.ndarray): Training images.
        Y (np.ndarray): Training masks.
        X_val (np.ndarray): Validation images.
        Y_val (np.ndarray): Validation masks.
        augment_configs (list): List of augmentation configurations.
        build_model_fn (function): Function to build the model.
        num_versions (int): Number of augmented versions per image.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        dict: Results of training with different configurations.
    """
    results = {}
    for i, config in enumerate(augment_configs):
        print(f"\nApplying Augmentation Configuration {i+1}/{len(augment_configs)}: {config}")
        X_aug, Y_aug = augment_dataset(X, Y, config, num_versions=num_versions)

        model = build_model_fn(X_aug.shape[1:])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_aug, Y_aug,
            validation_data=(X_val, Y_val),
            batch_size=batch_size,
            epochs=epochs
        )

        results[f"Config_{i+1}"] = history

        # Optional: Visualize predictions (can be customized for your dataset)
        # visualize_prediction(model, X_val, Y_val)

    return results
