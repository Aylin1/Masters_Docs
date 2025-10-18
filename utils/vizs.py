import matplotlib.pyplot as plt
import numpy as np
import laspy

def visualize_lidar_points(las_file, mode='3D', num_points=100000):
    """
    Visualize LiDAR point cloud data in 2D or 3D.
    
    Args:
        las_file (str): Path to the LAS/LAZ file.
        mode (str): '2D' for top-down view, '3D' for full point cloud visualization.
        num_points (int): Number of points to visualize (default = 100,000).
    """
    # Load LiDAR data
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T


    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if mode == '2D':
        plt.figure(figsize=(20,16))
        plt.scatter(x, y, c=z, cmap='terrain', s=0.5, alpha=0.5)
        plt.colorbar(label="Elevation (m)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("LiDAR Point Cloud (Top-Down 2D View)")
        plt.axis("equal")
        plt.show()

    elif mode == '3D':
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='terrain', s=0.5, alpha=0.5)

        # Labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Elevation (m)")
        ax.set_title("LiDAR Point Cloud (3D Visualization)")

        plt.show()
    else:
        raise ValueError("Invalid mode. Choose '2D' or '3D'.")



def visualize_layer_list(layers, layer_titles=None, mode='3D'):
    """
    Visualize raster layers (e.g., DEM, CHM, Slope, Aspect) in 3D with
    fixed colormaps and publication-style layout.

    Args:
        layers (list[np.ndarray]): List of 2D raster layers (each shape: [H, W]).
        layer_titles (list[str]): List of titles corresponding to layers.
        mode (str): '2D' or '3D'. Default = '3D' for terrain visualization.
    """
    num_layers = len(layers)
    if layer_titles is None:
        layer_titles = [f"Layer {i+1}" for i in range(num_layers)]

    # --- Fixed colormaps for each known layer ---
    cmap_dict = {
        "dem": ("terrain", "Elevation (m)", (np.nanmin(layers[0]), np.nanmax(layers[0]))),
        "chm": ("Greens", "Canopy Height (m)", (0, np.nanpercentile(layers[1], 99))),
        "slope": ("viridis", "Slope (°)", (0, 60)),
        "aspect": ("twilight", "Aspect (°)", (0, 360))
    }

    # --- Visualization mode ---
    if mode == '3D':
        fig = plt.figure(figsize=(5.5 * num_layers, 6))
        for i, (layer, title) in enumerate(zip(layers, layer_titles)):
            ax = fig.add_subplot(1, num_layers, i + 1, projection='3d')

            x = np.arange(layer.shape[1])
            y = np.arange(layer.shape[0])
            X, Y = np.meshgrid(x, y)

            key = title.lower()
            cmap, label, (vmin, vmax) = cmap_dict.get(key, ("Spectral", "Value", (np.nanmin(layer), np.nanmax(layer))))

            # Surface plot with proper lighting and elevation scale
            surf = ax.plot_surface(
                X, Y, layer,
                cmap=cmap,
                linewidth=0,
                antialiased=False,
                vmin=vmin,
                vmax=vmax,
                shade=True
            )

            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel(label)
            ax.tick_params(axis='both', which='major', labelsize=8)
            fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.05, label=label)

        plt.tight_layout()
        plt.show()

    elif mode == '2D':
        fig, axs = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
        if num_layers == 1:
            axs = [axs]

        for i, (layer, title) in enumerate(zip(layers, layer_titles)):
            key = title.lower()
            cmap, label, (vmin, vmax) = cmap_dict.get(key, ("Spectral", "Value", (np.nanmin(layer), np.nanmax(layer))))
            im = axs[i].imshow(layer, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[i].set_title(title, fontsize=12)
            axs[i].axis('off')
            cbar = plt.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
            cbar.set_label(label, fontsize=10)

        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Invalid mode. Choose '2D' or '3D'.")



# -----------------------------
# Overlay Visualization
# -----------------------------
def visualize_overlay(rgb_image, mask, alpha=0.4):
    """
    Overlay a ground truth mask on top of an RGB or grayscale image.

    Args:
        rgb_image (np.ndarray): Image array (H, W, 3).
        mask (np.ndarray): Ground truth mask array (H, W).
        alpha (float): Transparency level for mask overlay.
    """
    if rgb_image.ndim == 3 and rgb_image.shape[0] != rgb_image.shape[1]:
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # Convert from (C,H,W) → (H,W,C)

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image)
    plt.imshow(mask, cmap='Reds', alpha=alpha)
    plt.title("Ground Truth Overlay")
    plt.axis('off')
    plt.show()

def visualize_predictions(optimized_models_results, num_samples=4, 
                          optimal_thresholds=None, apply_optimal_thresholds=False, 
                          figsize=(10, 8)):
    """
    Visualize input images, ground truths, and predictions for multiple models.

    Parameters:
        optimized_models_results (dict): Dictionary with model names as keys.
            Each entry should contain a dictionary with key "predicted_images", which can be either:
                - A list of dictionaries with keys "input_image", "ground_truth", and "predicted_mask".
                - A dictionary with keys "input_image", "ground_truth", and "predicted_mask",
                  where each value is an array-like structure (with the sample index as one dimension).
        num_samples (int): Number of random samples to display. Default is 4.
        optimal_thresholds (dict, optional): Dictionary mapping model names to threshold values
            for binarizing predictions. If None, no thresholds are applied.
        apply_optimal_thresholds (bool): If True and `optimal_thresholds` is provided, the predicted
            masks are binarized using the corresponding threshold. Default is False.
        figsize (tuple): Figure size for the plot. Default is (10, 8).

    Returns:
        None. The function displays a matplotlib figure.
    """
    # Determine the structure and available sample count using the first model's results.
    first_model_key = list(optimized_models_results.keys())[0]
    predicted_images = optimized_models_results[first_model_key]["predicted_images"]

    if isinstance(predicted_images, dict):
        # Assuming all keys have the same length, get length from the first key.
        num_available_samples = len(next(iter(predicted_images.values())))
    else:
        num_available_samples = len(predicted_images)

    # Ensure we do not request more samples than available.
    num_samples = min(num_samples, num_available_samples)
    sample_indices = np.random.choice(num_available_samples, num_samples, replace=False)

    # Get the list of model names.
    models = list(optimized_models_results.keys())

    # Create a figure with subplots:
    #   - First column: Input Image (RGB channels only)
    #   - Second column: Ground Truth
    #   - One column per model's prediction
    num_columns = 2 + len(models)
    fig, axes = plt.subplots(num_samples, num_columns, figsize=figsize)

    # If only one sample is selected, ensure axes is 2D.
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # Iterate over each selected sample.
    for row_idx, sample_idx in enumerate(sample_indices):
        # Retrieve the input image and ground truth based on the structure.
        if isinstance(predicted_images, list):
            sample_data = predicted_images[sample_idx]
            input_image = sample_data["input_image"]
            ground_truth = sample_data["ground_truth"]
        else:
            input_image = predicted_images["input_image"][sample_idx]
            ground_truth = predicted_images["ground_truth"][sample_idx]

        # Plot the input image (selecting only the first three channels for RGB).
        axes[row_idx, 0].imshow(input_image[:, :, :3])
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title("Input Image", fontsize=8)

        # Plot the ground truth image.
        axes[row_idx, 1].imshow(ground_truth, cmap="gray")
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title("Ground Truth", fontsize=8)

        # Iterate over each model to plot its prediction.
        for col_idx, model_name in enumerate(models, start=2):
            model_predicted_images = optimized_models_results[model_name]["predicted_images"]
            if isinstance(model_predicted_images, list):
                predicted_mask = model_predicted_images[sample_idx]["predicted_mask"]
            else:
                predicted_mask = model_predicted_images["predicted_mask"][sample_idx]

            # Apply thresholding if requested and if a threshold is provided.
            if apply_optimal_thresholds and optimal_thresholds is not None and model_name in optimal_thresholds:
                threshold = optimal_thresholds[model_name]
                display_mask = (predicted_mask > threshold).astype(int)
            else:
                display_mask = predicted_mask

            axes[row_idx, col_idx].imshow(display_mask, cmap="gray")
            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(model_name, fontsize=8)

    plt.tight_layout()
    plt.show()

def visualize_lidar_points(las_file, mode='3D', num_points=100000):
    """
    Visualize LiDAR point cloud data in 2D or 3D.
    
    Args:
        las_file (str): Path to the LAS/LAZ file.
        mode (str): '2D' for top-down view, '3D' for full point cloud visualization.
        num_points (int): Number of points to visualize (default = 100,000).
    """
    # Load LiDAR data
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T


    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    if mode == '2D':
        plt.figure(figsize=(20,16))
        plt.scatter(x, y, c=z, cmap='terrain', s=0.5, alpha=0.5)
        plt.colorbar(label="Elevation (m)")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("LiDAR Point Cloud (Top-Down 2D View)")
        plt.axis("equal")
        plt.show()

    elif mode == '3D':
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=z, cmap='terrain', s=0.5, alpha=0.5)

        # Labels and title
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_zlabel("Elevation (m)")
        ax.set_title("LiDAR Point Cloud (3D Visualization)")

        plt.show()
    else:
        raise ValueError("Invalid mode. Choose '2D' or '3D'.")


def visualize_features_per_file(file_paths, subsample_ratio=0.1):
    """
    Load LAZ files, subsample, and visualize features for each file.
    """
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        las = laspy.read(file_path)
        num_points = len(las.x)
        sample_size = int(num_points * subsample_ratio)
        indices = np.random.choice(num_points, sample_size, replace=False)

        # Extract features
        points = np.vstack((las.x[indices], las.y[indices], las.z[indices])).T
        intensity = las.intensity[indices]
        classification = las.classification[indices]

        # Normalize height for visualization
        height = points[:, 2]
        normalized_height = (height - height.min()) / (height.max() - height.min())

        # Visualize histograms
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Intensity Histogram (Logarithmic)
        axes[0].hist(np.log1p(intensity), bins=50, color='blue', alpha=0.7)
        axes[0].set_title(f"Log-Scaled Intensity Histogram\n{file_path}")
        axes[0].set_xlabel("Log(Intensity)")
        axes[0].set_ylabel("Count")

        # Height Histogram
        axes[1].hist(height, bins=50, color='green', alpha=0.7)
        axes[1].set_title(f"Height Histogram\n{file_path}")
        axes[1].set_xlabel("Height")
        axes[1].set_ylabel("Count")

        # Classification Histogram
        axes[2].hist(classification, bins=np.arange(classification.min(), classification.max() + 2) - 0.5,
                     color='orange', alpha=0.7)
        axes[2].set_title(f"Classification Histogram\n{file_path}")
        axes[2].set_xlabel("Class ID")
        axes[2].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

def visualize_classification_masks(data):
    """
    Visualize all unique classification classes in the LiDAR data one by one as individual masks.

    Parameters:
        data (dict): LiDAR data containing points and classification feature.
    """
    points = data['points']
    x, y = points[:, 0], points[:, 1]
    classification = data['classification']

    # Get unique classification values
    unique_classes = np.unique(classification)

    for cls in unique_classes:
        # Mask for the current classification
        mask = classification == cls
        x_cls = x[mask]
        y_cls = y[mask]

        # Create scatter plot for the current class
        plt.figure(figsize=(10, 8))
        plt.scatter(x_cls, y_cls, color='blue', s=1, label=f'Class {cls}')
        plt.title(f"LiDAR Classification Mask for Class {cls}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()