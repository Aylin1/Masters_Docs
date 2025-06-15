This repository contains documentation, code references, and supplementary materials from my Master's thesis submitted to HTW Berlin, titled:

**"Evaluating the Impact of Data Augmentation on Forest Segmentation Using Aerial Imagery and LiDAR Data"**

## 📘 Project Overview

The research investigates how various data augmentation strategies and the integration of multimodal geospatial data (RGBI orthophotos and LiDAR point clouds) affect the performance of deep learning-based forest segmentation models. The study compares U-Net, U-Net-HRNet, and U-Net-FusionNet architectures, focusing on:

- **Segmentation accuracy under class imbalance**
- **Influence of seasonal and spectral discrepancies**
- **Effectiveness of structural features (CHM, slope, aspect)**
- **Augmentation strategies (spatial, spectral, noise-based)**

## 🧠 Key Contributions

- Developed a structured preprocessing pipeline for LiDAR and DOP data
- Generated DEM, DSM, CHM, slope, and aspect rasters
- Explored thresholding methods for binary segmentation masks
- Evaluated model performance using metrics like IoU and F1-score
- Analyzed the role of tree visibility and species density across modalities


> **Note**: Raw datasets are not included due to size and licensing. Please contact for access or reproduction instructions.

## 📌 Highlights

- Preprocessing aligned with ALS-DOP metadata for accurate georeferencing
- Integration of voxel-based interpolation and grid rasterization
- Comparative analysis of augmentation techniques (rotation, scaling, spectral shifts)
- Addressed seasonal mismatch in RGB and CHM tree visibility

## 🛠 Technologies Used

- **Languages**: Python, LaTeX
- **Libraries**: `laspy`, `rasterio`, `numpy`, `scikit-image`, `matplotlib`, `TensorFlow`
- **Tools**: QGIS, PDAL, Jupyter, VSCode, Git

**"Remote sensing and deep learning can together uncover the invisible layers of our forests."** 🌲

