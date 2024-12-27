import os
import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt

def find_rgbi_image(laz_data_path, dop_folder):
    file_identifier = os.path.basename(laz_data_path).split('.')[0].split('_')[-1]
    for root, _, files in os.walk(dop_folder):
        for file in files:
            if file_identifier in file and file.endswith('.tif'):
                return os.path.join(root, file)
    return None

def get_als_dop_matches(als_folder, dop_folder):
    matches = {}
    for root, _, files in os.walk(als_folder):
        for file in files:
            if file.endswith('.laz'):
                als_file_path = os.path.join(root, file)
                dop_image_path = find_rgbi_image(als_file_path, dop_folder)
                if dop_image_path:
                    matches[als_file_path] = dop_image_path
    return matches

def preprocess_lidar(file_path, voxel_size=1.0):
    """
    Process LiDAR data to generate DSM, DEM, and CHM with improved logic.
    """
    las = laspy.read(file_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # Extract ground points (class 2 in LAS classification)
    ground_mask = (las.classification == 2)
    ground_points = points[ground_mask]

    # Define grid
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    x_grid = np.arange(x_min, x_max, voxel_size)
    y_grid = np.arange(y_min, y_max, voxel_size)
    grid_x, grid_y = np.meshgrid(x_grid, y_grid)

    # DEM: Interpolate ground points onto the grid
    dem = griddata(
        ground_points[:, :2],
        ground_points[:, 2],
        (grid_x, grid_y),
        method="linear"
    )

    # DSM: Use the maximum z-value within each grid cell
    dsm = np.full(grid_x.shape, np.nan)
    for point in points:
        x_idx = int((point[0] - x_min) / voxel_size)
        y_idx = int((point[1] - y_min) / voxel_size)
        if np.isnan(dsm[y_idx, x_idx]):
            dsm[y_idx, x_idx] = point[2]
        else:
            dsm[y_idx, x_idx] = max(dsm[y_idx, x_idx], point[2])

    # CHM: DSM - DEM
    chm = dsm - dem
    chm[chm < 0] = 0  # Ensure non-negative values

    return grid_x, grid_y, dem, dsm, chm, x_min, y_max, voxel_size

def compute_slope_aspect(dem, grid_resolution):
    """
    Compute slope and aspect from DEM.
    """
    dz_dx, dz_dy = np.gradient(dem, grid_resolution)
    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)) * (180 / np.pi)
    aspect = (np.arctan2(-dz_dy, dz_dx) * (180 / np.pi) + 360) % 360
    return slope, aspect

def visualize_raster_layers(dem, dsm, chm, slope, title_prefix):
    """
    Visualize DEM, DSM, CHM, and slope layers.
    """
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(dem, cmap='terrain')
    axs[0].set_title(f"{title_prefix} - DEM")
    axs[1].imshow(dsm, cmap='terrain')
    axs[1].set_title(f"{title_prefix} - DSM")
    axs[2].imshow(chm, cmap='terrain')
    axs[2].set_title(f"{title_prefix} - CHM")
    axs[3].imshow(slope, cmap='viridis')
    axs[3].set_title(f"{title_prefix} - Slope")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def save_as_geotiff(output_path, data, x_min, y_max, resolution, crs="EPSG:32633"):
    """
    Save raster data as a GeoTIFF file.
    """
    transform = from_origin(x_min, y_max, resolution, resolution)
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype='float32', crs=crs, transform=transform
    ) as dst:
        dst.write(data, 1)
    print(f"Saved: {output_path}")

def process_and_save(als_folder, dop_folder, output_base_dir):
    matches = get_als_dop_matches(als_folder, dop_folder)
    print(f"Found {len(matches)} ALS-DOP pairs.")
    
    for als_file, dop_image in matches.items():
        print(f"Processing ALS: {als_file}, DOP: {dop_image}")
        base_name = os.path.basename(als_file).split('.')[0]
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Process LiDAR data
        grid_x, grid_y, dem, dsm, chm, x_min, y_max, resolution = preprocess_lidar(als_file)
        slope, _ = compute_slope_aspect(dem, resolution)  # Aspect not required for this setup

        # Flip rasters vertically to align with the reference DOP image
        dem = np.flipud(dem)
        dsm = np.flipud(dsm)
        chm = np.flipud(chm)
        slope = np.flipud(slope)

        # Visualize calculated layers
        #visualize_raster_layers(dem, dsm, chm, slope, title_prefix=base_name)

        # Save LiDAR derivatives as GeoTIFFs aligned with the DOP image's metadata
        save_as_geotiff_with_reference(
            os.path.join(output_dir, f"{base_name}_dem.tif"), dem, dop_image
        )
        save_as_geotiff_with_reference(
            os.path.join(output_dir, f"{base_name}_chm.tif"), chm, dop_image
        )
        save_as_geotiff_with_reference(
            os.path.join(output_dir, f"{base_name}_slope.tif"), slope, dop_image
        )

def save_as_geotiff_with_reference(output_path, data, reference_image):
    """
    Save raster data as a GeoTIFF file aligned with the reference image's geospatial metadata.
    """
    with rasterio.open(reference_image) as ref:
        transform = ref.transform
        crs = ref.crs

    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype='float32', crs=crs, transform=transform
    ) as dst:
        dst.write(data, 1)
    print(f"Saved: {output_path}")


def stack_and_save(als_folder, dop_folder, output_base_dir):
    matches = get_als_dop_matches(als_folder, dop_folder)
    print(f"Found {len(matches)} ALS-DOP pairs.")

    for als_file, dop_image in matches.items():
        print(f"Processing ALS: {als_file}, DOP: {dop_image}")
        base_name = os.path.basename(als_file).split('.')[0]
        output_dir = os.path.join(output_base_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)



        # Create stacked raster
        create_stacked_raster(
            output_dir=output_dir,
            base_name=base_name,
            dop_image=dop_image,
            chm_path=os.path.join(output_dir, f"{base_name}_chm.tif"),
            slope_path=os.path.join(output_dir, f"{base_name}_slope.tif")
        )

        print(f"Processed and saved outputs for {base_name}.")


def create_stacked_raster(output_dir, base_name, dop_image, chm_path, slope_path):
    stacked_raster_path = os.path.join(output_dir, f"{base_name}_final_input.tif")

    # Read metadata from the DOP file
    with rasterio.open(dop_image) as src:
        meta = src.meta.copy()
        meta.update(count=6)  # 4 bands from DOP + 3 from LiDAR layers

    # Write stacked raster
    with rasterio.open(stacked_raster_path, "w", **meta) as dst:
        # Add 4 bands from DOP
        with rasterio.open(dop_image) as dop_src:
            for i in range(1, 5):  # Bands 1 to 4
                dst.write_band(i, dop_src.read(i))

        # Add DEM, CHM, and Slope as Bands 5-7
        for idx, raster_path in enumerate([chm_path, slope_path], start=5):
            with rasterio.open(raster_path) as src:
                dst.write_band(idx, src.read(1))

    print(f"Stacked raster saved at: {stacked_raster_path}")