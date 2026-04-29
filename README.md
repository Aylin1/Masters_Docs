# Evaluating the Impact of Data Augmentation on Forest Segmentation Using Aerial Imagery and LiDAR data

**Author:** Aylin Gülüm  
**Institution:** Hochschule für Technik und Wirtschaft Berlin (HTW Berlin)  
**Program:** M.Sc. Project Management & Data Science  
 

---

## Abstract

This thesis examines how data augmentation and multimodal feature integration influence the performance of deep learning models for forest segmentation using high-resolution aerial imagery and LiDAR data. The study integrates RGBI orthophotos with LiDAR-derived structural layers—including canopy height, elevation, slope, and aspect—to evaluate how spectral and structural information jointly contribute to more accurate and generalizable segmentation outcomes. A standardized preprocessing pipeline was developed to align and fuse these heterogeneous datasets and to generate ground-truth masks using a hybrid SAM-assisted and manually refined approach.

Three segmentation architectures **U-Net**, **U-Net-HRNet**, and **U-Net-FusionNet** trained under a range of augmentation strategies to assess their sensitivity to multiscale vegetation patterns and spatial heterogeneity.
Three architectures were compared under various augmentation strategies to assess performance on high-resolution forest segmentation tasks.

**Key contributions:**
- Comparison of multiple model architectures for forest segmentation  
- Analysis of augmentation strategies and multimodal data fusion  
- Reproducible pipeline for preprocessing, training, and evaluation


## Repository Structure

masters_docs/

notebooks/                     # Jupyter notebooks for exploration and preprocessing
-    ├── Experiments.ipynb         # Experiment notebooks with model evaluation
-    ├── exploration_las_files.ipynb  # Exploration of LiDAR LAS files
-    ├── preprocess_data.ipynb    # Data preprocessing pipeline
-    └── samgeo_ground_truth.ipynb # Ground-truth mask exploration

utils/                         # Utility scripts for augmentation and preprocessing
-    ├── augmentation_pipeline.py  # Data augmentation functions
-    ├── get_file_matches.py       # Helper for file matching
-    ├── load_train_eval.py            # Script for loading, training, and evaluating models
-    ├──    models.py  # Deep learning models (Unet and derivatives)
-    ├── preprocess_and_stack.py   # Functions for preprocessing and stacking inputs

README.md                        # Project documentation

requirements.txt                 # Python dependencies

---

## Methodology

### Workflow Diagram

Below is the complete workflow summarizing data acquisition, preprocessing, feature extraction, model training, and evaluation.

<p align="center"> <img src="visuals/workflow_1.png" alt="Workflow Diagram" width="750"> </p>


## Dataset and Methods

### 1. **Study Area and Data Sources**

### Dataset

**Region:** Tschernitz, Brandenburg (Germany)  
**Data Source:** [Landesvermessung und Geobasisinformation Brandenburg (LGB)](https://geobasis-bb.de)  
**Portal:** [GeoPortal Brandenburg – Open Data](https://geoportal.brandenburg.de)  
**License:** *Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0)*  


   - Study region: Tschernitz (1 km²), eastern Germany  
   - RGBI orthophotos provided by Geobasis Brandenburg (TrueDOP)  
   - LiDAR point clouds (Airborne Laser Scanning, LAS 1.4, Point Format 6)  
   - RGBI specifications:  
     - Spatial resolution: 0.2 m × 0.2 m  
     - Spectral bands: Red, Green, Blue, Infrared  
     - Acquisition date: 07 April 2024  
     - Publication date: 09 August 2024  
   - LiDAR specifications:  
     - Point density: 5 points/m²  
     - Acquisition date: 08 January 2023  
     - Publication date: 28 November 2023  
     - Attributes include spatial coordinates, intensity, classification, return information, scan angle, and GPS time  
   - The ~455-day temporal difference between RGBI and LiDAR is acceptable due to stable forest canopy structure in the region.

### 2. **Data Preprocessing & Feature Extraction**  
   - Rasterization of raw LiDAR point clouds without point removal  
   - Generation of:  
     - Digital Elevation Model (DEM)  
     - Canopy Height Model (CHM)  
     - Slope  
     - Aspect  
   - Initial rasterization at 1 m resolution due to memory limitations  
   - Bicubic upscaling of LiDAR-derived rasters to 5000 × 5000 pixels to match RGBI orthophotos  
   - Co-alignment and merging of RGBI and LiDAR layers into 20 multimodal raster stacks (each representing a 1 km² tile)

<p align="center">
  <img src="visuals/spectral_structural_insights.png" alt="Spectral and Structural Insights" width="750">
</p>

Differences between spectral and structural representations reveal important characteristics of forest composition.  In RGB imagery, certain tree species or sparse canopy structures may appear faint or visually ambiguous. The Canopy Height Model (CHM), however, clearly highlights these same trees due to their elevation and structural form. This divergence indicates that relying solely on spectral information may lead to under-segmentation of tall or sparsely foliated trees, while the integration of LiDAR height data improves separability.


### 3. **Ground-Truth Label Generation**  
   - Use of the Segment Anything Model (SAM) with a prompt-based approach  
   - Orthophotos divided into 1024 × 1024 px patches and processed with the prompt “forest”  
   - Adjustment of SAM box/text thresholds to refine segmentation quality  
   - Manual correction of tiling artifacts using GIMP  
   - Final masks compiled into full-tile ground-truth segmentation layers

### 4. **Data Partitioning, Spatial Structure, and Class Balance**  
   - Each multimodal tile (5000 × 5000 px) resized to 256 × 256 px (≈3.9 m per pixel)  
   - Full-tile resizing used instead of patch-based tiling to preserve spatial continuity  
   - Spatial holdout split used to prevent geographic leakage:  
     - 80% training tiles  
     - 20% validation tiles (geographically distinct subset)  
   - Stratified variant evaluated due to forest-cover imbalance:  
     - Original forest coverage: 30.5% (train), 53.0% (val)  
     - After stratification: 31.0% (train), 50.6% (val)  
   - Combined Dice + Binary Cross-Entropy loss applied to mitigate imbalance during training

### 5. **Model Architectures**  
   - Three U-Net–based convolutional neural network architectures were implemented to evaluate the impact of structural and spectral feature integration.  
   - **Baseline U-Net:**  
     - Lightweight variant with reduced convolutional filters  
     - Depthwise-separable convolutions for efficiency  
     - Standard encoder–decoder structure with skip connections  
   - **U-Net-HRNet (HRNet-Inspired):**  
     - Incorporates high-resolution feature retention principles  
     - Integrates Squeeze-and-Excitation (SE) blocks for channel-wise attention  
     - Designed to preserve fine spatial detail lost during downsampling  
   - **U-Net-FusionNet (FusionNet-Inspired):**  
     - Includes residual connections in both encoder and decoder  
     - Enhances gradient flow and multimodal feature propagation  
     - Improves stability when integrating RGBI and LiDAR-derived features  
   - All models were trained using aligned, multimodal raster stacks and evaluated consistently across splits.

---

### 6. **Augmentation Strategy**  
   - Offline data augmentation used to expand the diversity of training samples.  
   - Augmentation types included:  
     - Geometric transformations (rotation, horizontal/vertical flips, random scaling)  
     - Photometric adjustments (brightness and contrast shifts)  
     - Noise-based augmentations (Gaussian noise, salt-and-pepper noise)  
     - Spectral dropout applied selectively to multispectral channels  
   - Augmentations were parameterized to avoid unrealistic forest representations while improving model generalization.

---

### 7. **Evaluation Metrics**  
   - Performance was assessed using commonly adopted segmentation metrics:  
     - **Intersection over Union (IoU):** Measures overlap between predicted and true masks.  
     - **Dice Coefficient:** Favors foreground classes and balances precision–recall interactions.  
     - **Precision and Recall:** Evaluate omission and commission errors in forest detection.  
   - Validation results were computed across spatially distinct tiles to ensure geographic generalization.

---

### 8. **Limitations**  
   - Temporal differences between RGBI and LiDAR datasets (~455 days) may introduce subtle inconsistencies despite stable canopy conditions.  
   - Upscaling LiDAR rasters to match RGBI resolution can smooth fine structural variations.  
   - Ground-truth masks generated via SAM required manual refinement to correct tiling artifacts.  
   - Forest coverage imbalance between tiles required stratification and loss-function adjustments.  
   - Performance may vary across forest types or seasons not represented in the dataset.

---

### 9. **Results Overview**  
   - The integration of LiDAR-derived structural features improved segmentation performance relative to RGBI-only models.  
   - Models incorporating residual connections (FusionNet-inspired U-Net) demonstrated improved stability and better multimodal fusion.  
   - HRNet-inspired U-Net improved detection of fine spatial detail and reduced boundary ambiguity.  
   - Augmentation strategies contributed significantly to model robustness, especially for mixed-species forest areas.  
   - Spatial holdout evaluation confirmed that multimodal fusion enhances generalization across distinct geographic tiles.


## Installation and Reproduction
This project was developed with Python 3.10 and TensorFlow 2.10.

Presentation of this thesis is available in the file:  
[**thesis_presentation.pdf**](visuals/thesis_presentation.pdf)
