import os
import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
import tensorflow as tf
import os
from get_file_matches import get_als_dop_matches
import time

def preprocess_lidar(file_path, voxel_size):
    """
    Process LiDAR data to generate DSM, DEM, and CHM with timing for each step.

    Args:
        file_path (str): Path to the .laz (LiDAR) file.
        voxel_size (float): Desired grid spacing in X and Y for rasterization
                            (same as orthoimage resolution).
    
    Returns:
        grid_x, grid_y, dem, dsm, chm, x_min, y_max, voxel_size
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    timings = {}  # Dictionary to store durations for each step

    # Extract ground points (class 2 in LAS classification)
    start_time = time.time()
    ground_mask = (las.classification == 2)
    ground_points = points[ground_mask]
    timings["Extract Ground Points"] = time.time() - start_time

    # Define grid in X and Y
    start_time = time.time()
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_grid = np.arange(x_min, x_max, voxel_size)
    y_grid = np.arange(y_min, y_max, voxel_size)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    # DEM: Interpolate ground points onto the grid (linear interpolation)
    start_time = time.time()
    dem = griddata(
        ground_points[:, :2],
        ground_points[:, 2],
        (grid_x, grid_y),
        method="linear"
    )
    timings["Calculate DEM"] = time.time() - start_time

    # DSM: Use the maximum z-value within each cell
    start_time = time.time()
    dsm = np.full(grid_x.shape, np.nan)
    for point in points:
        # Convert the point’s (x, y) into grid indices
        x_idx = int((point[0] - x_min) / voxel_size)
        y_idx = int((point[1] - y_min) / voxel_size)
        if 0 <= x_idx < dsm.shape[1] and 0 <= y_idx < dsm.shape[0]:
            if np.isnan(dsm[y_idx, x_idx]):
                dsm[y_idx, x_idx] = point[2]
            else:
                dsm[y_idx, x_idx] = max(dsm[y_idx, x_idx], point[2])

    # CHM: DSM - DEM (clip negative values to zero)
    start_time = time.time()
    chm = dsm - dem
    chm[chm < 0] = 0

    slope, aspect = compute_slope_aspect(dem, voxel_size)

    # Calculate total and average time
    total_time = sum(timings.values())
    avg_time = total_time / len(timings)

    print("==== Timing Report ====")
    for feature, duration in timings.items():
        print(f"{feature}: {duration:.4f} seconds")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Time: {avg_time:.4f} seconds")

    return dem, dsm, chm, slope, aspect



def compute_slope_aspect(dem, cell_size):
    """
    Compute slope and aspect from DEM.

    Args:
        dem (ndarray): 2D array representing the DEM.
        cell_size (float): Spacing of the grid cells in X and Y.
    
    Returns:
        slope (ndarray): Slope (in degrees).
        aspect (ndarray): Aspect (in degrees, 0-360).
    """
    dz_dx, dz_dy = np.gradient(dem, cell_size)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
    aspect = (np.arctan2(-dz_dy, dz_dx) * (180 / np.pi) + 360) % 360
    return slope, aspect


def process_and_save(als_folder, dop_folder, output_base_dir, voxel_size=1):
    """
    For each matched ALS (.laz) - DOP (orthophoto) pair, read the DOP resolution,
    preprocess LiDAR using that resolution, compute DEM/DSM/CHM, slope, and save
    each as a GeoTIFF aligned with the DOP's metadata.
    """
    matches = get_als_dop_matches(als_folder, dop_folder)
    print(f"Found {len(matches)} ALS-DOP pairs.")

    for als_file, dop_image in matches.items():
        print(f"Processing ALS: {als_file}\nDOP: {dop_image}")
        base_name = os.path.basename(als_file).split('.')[0]
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # -- Preprocess LiDAR at the same resolution
        dem, dsm, chm, slope, aspect = preprocess_lidar(als_file, voxel_size=voxel_size)

        # -- Flip arrays if needed to match top-down orientation
        # Check if the DOP transform suggests top-left origin vs. bottom-left origin:
        # If the yres is negative, typical for North-up, the array might need flipping to match DOP orientation.
        # This step is optional depending on how you'd like to align them.
        dem = np.flipud(dem)
        dsm = np.flipud(dsm)
        chm = np.flipud(chm)
        slope = np.flipud(slope)
        aspect = np.flipud(aspect)

        # -- Save LiDAR derivatives as GeoTIFFs aligned with the DOP
        dem_path = os.path.join(output_dir, f"{base_name}_dem.tif")
        chm_path = os.path.join(output_dir, f"{base_name}_chm.tif")
        slope_path = os.path.join(output_dir, f"{base_name}_slope.tif")
        aspect_path = os.path.join(output_dir, f"{base_name}_aspect.tif")

        save_as_geotiff_with_reference(dem_path, dem, dop_image)
        save_as_geotiff_with_reference(chm_path, chm, dop_image)
        save_as_geotiff_with_reference(slope_path, slope, dop_image)
        save_as_geotiff_with_reference(aspect_path, aspect, dop_image)
        
        print(f"Saved DEM, CHM, and slope to {output_dir}.\n")


def save_as_geotiff_with_reference(output_path, data, reference_image):
    """
    Save a 2D array as a GeoTIFF aligned with the reference image's metadata.

    Args:
        output_path (str): Path to output .tif file.
        data (ndarray): 2D array to be saved.
        reference_image (str): Path to an existing GeoTIFF whose metadata we want to replicate (CRS, transform).
    """
    with rasterio.open(reference_image) as ref:
        transform = ref.transform
        crs = ref.crs

        # NOTE: We use the shape of 'data', so must ensure it matches the reference image dimensions.
        # If there's a mismatch, consider interpolation or resizing.
        out_height, out_width = data.shape

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=out_height, width=out_width,
        count=1, dtype='float32', crs=crs, transform=transform
    ) as dst:
        dst.write(data, 1)
    print(f"Saved: {output_path}")


def stack_and_save(als_folder, dop_folder, output_base_dir):
    """
    For each matched ALS-DOP pair, stack the (4-band) DOP with LiDAR-derived
    layers (CHM, slope, aspect) into a single multi-band GeoTIFF.
    """
    matches = get_als_dop_matches(als_folder, dop_folder)
    print(f"Found {len(matches)} ALS-DOP pairs.")

    for als_file, dop_image in matches.items():
        print(f"Processing ALS: {als_file}\nDOP: {dop_image}")
        base_name = os.path.basename(als_file).split('.')[0]
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Paths to LiDAR-derived rasters
        chm_path = os.path.join(output_dir, f"{base_name}_chm.tif")
        slope_path = os.path.join(output_dir, f"{base_name}_slope.tif")
        aspect_path = os.path.join(output_dir, f"{base_name}_aspect.tif")

        # Create stacked raster with resampling
        create_stacked_raster(
            output_dir=output_dir,
            base_name=base_name,
            dop_image=dop_image,
            chm_path=chm_path,
            slope_path=slope_path,
            aspect_path=aspect_path
        )
        print(f"Processed and saved outputs for {base_name}.\n")


def create_stacked_raster(output_dir, base_name, dop_image, chm_path, slope_path, aspect_path):
    """
    Combine the DOP (4-band) with LiDAR layers (CHM, slope, aspect) into one multi-band TIFF.
    Uses TensorFlow's tf.image.resize to reshape LiDAR layers to match the DOP resolution.
    """
    stacked_raster_path = os.path.join(output_dir, f"{base_name}_stacked_7bands.tif")

    with rasterio.open(dop_image) as dop_src:
        meta = dop_src.meta.copy()
        width, height = dop_src.width, dop_src.height
        transform = dop_src.transform
        meta.update(count=7)  # Total 7 bands (4 DOP + 3 LiDAR)

        # Read DOP (4-band)
        dop_data = dop_src.read()

    def reshape_raster(src_path, target_shape):
        """ Reshape raster using TensorFlow resize (bicubic for smooth scaling). """
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)  # Ensure float32 for TF operations
            data = np.expand_dims(data, axis=-1)  # Convert (H, W) → (H, W, 1)
            resized_data = tf.image.resize(data, target_shape, method="bicubic").numpy()
            return np.squeeze(resized_data)  # Remove extra channel dimension

    # Reshape CHM, Slope, and Aspect to match the DOP resolution
    target_shape = (height, width)  # Match DOP resolution
    chm_resized = reshape_raster(chm_path, target_shape)
    slope_resized = reshape_raster(slope_path, target_shape)
    aspect_resized = reshape_raster(aspect_path, target_shape)

    # Save Stacked Raster
    with rasterio.open(stacked_raster_path, "w", **meta) as dst:
        # Write DOP bands
        for i in range(1, 5):
            dst.write(dop_data[i - 1], i)

        # Write reshaped CHM, Slope, and Aspect
        dst.write(chm_resized, 5)
        dst.write(slope_resized, 6)
        dst.write(aspect_resized, 7)

    print(f"Stacked raster saved at: {stacked_raster_path}")


def tile_image(image, tile_size=(1024, 1024), pad=False):
    """
    Tiles an image into fixed-size patches, ensuring uniform shapes via padding.
    
    Args:
        image (numpy.ndarray): Input image (H, W, C) or (H, W) for grayscale.
        tile_size (tuple): Desired tile size (tile_height, tile_width).
        pad (bool): If True, pads the image to ensure all tiles are the same size.

    Returns:
        numpy.ndarray: Array of shape (num_tiles, tile_height, tile_width, C).
    """
    height, width = image.shape[:2]
    tile_height, tile_width = tile_size

    # If image is grayscale (H, W), add a channel dimension (H, W, 1)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]  

    num_channels = image.shape[2]

    # **Ensure padding to match tile size**
    if pad:
        pad_h = (tile_height - (height % tile_height)) % tile_height
        pad_w = (tile_width - (width % tile_width)) % tile_width
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    # Compute the new padded height and width
    new_height, new_width = image.shape[:2]
    tiles = []

    # Generate tiles
    for i in range(0, new_height, tile_height):
        for j in range(0, new_width, tile_width):
            tile = image[i:i+tile_height, j:j+tile_width]
            tiles.append(tile)

    return np.array(tiles)  # Now guaranteed to have consistent shapes


def load_raster_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
    return np.moveaxis(data, 0, -1)  # Move channel axis to the end


def load_data(matches, tile_size=(1250, 1250), resize_shape=(256, 256), use_tiling=False, save_tiles=False, save_dir="data/Tschernitz/tiles/train_test_tiled"):
    """
    Loads and processes dataset by tiling images and resizing each tile.

    Args:
        matches (dict): Dictionary mapping stacked TIFs to ground truth masks.
        tile_size (tuple): Size of tiles (height, width).
        resize_shape (tuple): Final shape for resizing (height, width).
        use_tiling (bool): If False, skips tiling and just resizes the whole image.
        save_tiles (bool): If True, saves tiled images to disk.
        save_dir (str): Directory to save the tiled images.

    Returns:
        np.ndarray, np.ndarray: Processed images and masks.
    """
    X, Y = [], []

    if save_tiles:
        os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    for stacked_tif, ground_truth_tif in matches.items():
        # Load data
        stacked = load_raster_data(stacked_tif)  # (H, W, C)
        ground_truth = load_raster_data(ground_truth_tif)  # (H, W) or (H, W, 1)

        # Extract file name for saving
        stacked_filename = os.path.basename(stacked_tif).replace(".tif", "")

        # Ensure ground truth mask has the correct shape
        if len(ground_truth.shape) == 2:
            ground_truth = ground_truth[..., np.newaxis]  # Convert (H, W) → (H, W, 1)

        if use_tiling:
            stacked_tiles = tile_image(stacked, tile_size)
            mask_tiles = tile_image(ground_truth, tile_size)

            for idx, (stacked_tile, mask_tile) in enumerate(zip(stacked_tiles, mask_tiles)):
                resized_stacked = tf.image.resize(stacked_tile, resize_shape, method="bicubic").numpy()
                resized_mask = tf.image.resize(mask_tile, resize_shape, method="nearest").numpy()

                # 🔹 Ensure all masks are (256, 256, 1)
                if resized_mask.ndim == 2:
                    resized_mask = resized_mask[..., np.newaxis]
                elif resized_mask.shape[-1] != 1:
                    resized_mask = resized_mask[..., 0:1]

                X.append(resized_stacked)
                Y.append(resized_mask)

                # 🔹 Save tiles if option is enabled
                if save_tiles:
                    tile_filename = f"{stacked_filename}_tile_{idx}.tif"
                    save_raster(os.path.join(save_dir, f"stacked_{tile_filename}"), resized_stacked)
                    save_raster(os.path.join(save_dir, f"mask_{tile_filename}"), resized_mask)
        else:
            # Resize whole image if tiling is disabled
            resized_stacked = tf.image.resize(stacked, resize_shape, method="bicubic").numpy()
            resized_mask = tf.image.resize(ground_truth, resize_shape, method="nearest").numpy()

            # 🔹 Ensure all masks are (256, 256, 1)
            if resized_mask.ndim == 2:
                resized_mask = resized_mask[..., np.newaxis]
            elif resized_mask.shape[-1] != 1:
                resized_mask = resized_mask[..., 0:1]

            X.append(resized_stacked)
            Y.append(resized_mask)

    # 🔹 Convert lists to NumPy arrays
    X = np.stack(X, axis=0)  # Shape (N, 256, 256, C)
    Y = np.stack(Y, axis=0)  # Shape (N, 256, 256, 1)
    
    return X, Y


def save_raster(output_path, data):
    """
    Saves raster data to a GeoTIFF file.
    
    Args:
        output_path (str): Path to save the raster.
        data (numpy.ndarray): Raster data (H, W, C) or (H, W).
    """
    height, width = data.shape[:2]
    num_bands = 1 if len(data.shape) == 2 else data.shape[2]

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=num_bands,
        dtype=data.dtype
    ) as dst:
        if num_bands == 1:
            dst.write(data.squeeze(), 1)  # Remove extra dimension
        else:
            for i in range(num_bands):
                dst.write(data[..., i], i + 1)
    print(f"Saved: {output_path}")

