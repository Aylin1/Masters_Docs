# Evaluating the Impact of Data Augmentation on Forest Segmentation Using Aerial Imagery and LiDAR Data

**Master's Thesis, HTW Berlin**
**Author:** Aylin Gülüm
**Institution:** Hochschule für Technik und Wirtschaft Berlin (HTW Berlin)
**Program:** M.Sc. Project Management and Data Science

---

## Abstract
This thesis examines how data augmentation and multimodal feature integration influence the performance of deep learning models for forest segmentation using high-resolution aerial imagery and LiDAR data. The study integrates RGBI orthophotos with LiDAR-derived structural layers—including canopy height, elevation, slope, and aspect—to evaluate how spectral and structural information jointly contribute to more accurate and generalizable segmentation outcomes. A standardized preprocessing pipeline was developed to align and fuse these heterogeneous datasets and to generate ground-truth masks using a hybrid SAM-assisted and manually refined approach.

Three segmentation architectures U-Net, U-Net-HRNet, and U-Net-FusionNet trained under a range of augmentation strategies to assess their sensitivity to multiscale vegetation patterns and spatial heterogeneity. Three architectures were compared under various augmentation strategies to assess performance on high-resolution forest segmentation tasks.

---

## Research Questions

### RQ1: Multimodal Contribution
**Does LiDAR-derived structural information improve segmentation quality compared to spectral-only methods?**

### RQ2: Architectural Sensitivity
**How do different U-Net-based architectures respond to multimodal fusion and hyperparameter tuning?**

### RQ3: Augmentation Robustness
**Which augmentation strategies improve generalization, and do they generalize across all architectures?**

**Key Findings:**
- LiDAR-derived structural features boost segmentation accuracy from 69.1% (RGB only) to 75.1% (RGBI+CHM), a 6.0 percentage point improvement
- Architecture design significantly influences augmentation sensitivity; no universal optimal augmentation strategy exists across all model types
- FusionNet with residual connections achieves superior performance (77.1% IoU) and stability compared to standard U-Net (75.3%) and HRNet variants (76.7%)

This work addresses critical gaps in multimodal remote sensing by quantifying how data fusion improves forest detection accuracy and how augmentation strategies must be tailored to specific architectural designs.
---

## Repository Structure

```
Masters_Docs/
├── README.md (this file)
├── notebooks/
│   ├── Experiments.ipynb                 # Comprehensive experiment notebook
│   ├── exploration_las_files.ipynb      # LiDAR data exploration
│   ├── preprocess_data.ipynb            # Data preprocessing pipeline
│   └── samgeo_ground_truth.ipynb        # Ground-truth mask generation
├── utils/
│   ├── augmentation_pipeline.py         # Data augmentation functions
│   ├── get_file_matches.py              # LiDAR-orthophoto file matching
│   ├── load_train_eval.py               # Training and evaluation workflow
│   ├── models.py                        # CNN architectures (U-Net variants)
│   └── preprocess_and_stack.py          # Feature extraction and stacking
├── visuals/
│   ├── workflow_1.png                   # Complete methodology workflow diagram
│   ├── spectral_structural_insights.png # Spectral vs. LiDAR feature comparison
│   ├── archs.drawio.png                 # Architecture diagrams
│   ├── test_loss.png                    # Learning rate sensitivity plot
│   ├── convergence.png                  # Training convergence curves
│   └── thesis_presentation.pdf          # Master's thesis defense slides
└── requirements.txt
```

---

## Experimental Design and Results

Below is the complete workflow summarizing data acquisition, preprocessing, feature extraction, model training, and evaluation.

<p align="left">
  <img src="visuals\workflow_1.png" alt="workflow_1" width="750">
</p>

### Study Area and Data

**Location:** Tschernitz, Brandenburg, Germany (1 km² area)

**RGBI Orthophotos:**
- Resolution: 0.2 m per pixel
- Dimensions: 5000×5000 pixels per tile
- Spectral bands: Red, Green, Blue, Infrared (RGBI)
- Acquisition date: April 7, 2024
- Source: Geobasis Brandenburg (TrueDOP product)

**LiDAR Point Clouds:**
- Density: 5 points per m²
- Format: LAS 1.4, Point Format 6
- Attributes: X, Y, Z coordinates, intensity, classification, return number, scan direction, GPS time
- Acquisition date: January 8, 2023
- Source: Geobasis Brandenburg

**Dataset:**
- 20 tiles (each 1 km²)
- 80% training / 20% validation (stratified by forest coverage density)
- Temporal offset: 455 days between RGBI and LiDAR (acceptable due to stable forest canopy structure)

---

### Preprocessing and Feature Extraction

**LiDAR Processing:**
1. Filtered by classification (ground returns, class 2) to isolate terrain
2. Rasterized at 1 m resolution to create:
   - Digital Elevation Model (DEM): interpolated ground surface
   - Digital Surface Model (DSM): maximum vegetation height
   - Canopy Height Model (CHM): DSM minus DEM
   - Slope: gradient of DEM
   - Aspect: orientation of slope
3. Bicubic upsampling from 1 m to 0.2 m resolution to match RGBI

**Raster Stacking:**
- RGBI + CHM (best combination): 5 channels
- Optional: RGBI + CHM + Slope/Aspect: 6-7 channels
- Resolution: 5000×5000 pixels per tile
- Data type: float32

**Ground-Truth Generation:**
- Zero-shot Segment Anything Model (SAM) with prompt "forest"
- Automatic thresholding and tiling artifact removal
- Manual refinement in GIMP to correct edge artifacts
- Final: Binary forest/non-forest masks

**Data Partitioning:**
- Full tiles resized to 256×256 pixels (3.9 m per pixel effective resolution)
- Stratified holdout: tiles assigned to train/val with forest-coverage stratification
- Prevents geographic leakage while maintaining class balance

---

### Experiment 1: Band Comparison (Multimodal Contribution)

- **Architecture:** Lightweight U-Net (baseline)
- **Training Setup:** Batch size 4, Learning rate 1e-3, 25 epochs, threshold 0.5

| Band Combination | IoU (%) | Dice (%) | Accuracy (%) | Gain vs. RGB |
|---|---|---|---|---|
| RGB | 69.1 | 75.5 | 91.89 | baseline |
| RGBI | 70.4 | 76.9 | 91.80 | +1.3 |
| RGBI + CHM | 75.1 | 81.8 | 93.87 | +6.0 |
| RGBI + CHM + Slope | 75.0 | 80.9 | 93.75 | -0.1 |
| All 7 Bands | 74.0 | 80.0 | 93.50 | -1.1 |

**Key Insights:**
- Adding NIR band provides modest improvement (+1.3 IoU)
- CHM integration yields significant performance boost (+6.0 IoU)
- Topographic features (slope, aspect) show diminishing returns
- All available bands underperform selective combination (information redundancy)

---

### Experiment 2: Model Architecture Comparison

- **Best Band Combination:** RGBI + CHM
- **Training Setup:** Batch size 4, Learning rate sweep 1e-2 to 1e-6, 50 epochs, Dice+BCE loss

**Architecture Details:**

<p align="left">
  <img src="visuals\archs.drawio.png" alt="Spectral and Structural Insights" width="900">
</p>


**Lightweight U-Net:**
- Depthwise-separable convolutions for efficiency
- 3 encoding levels with progressive feature doubling
- Standard skip connections

**U-Net-HRNet (with Squeeze-and-Excitation blocks):**
- 4 encoding levels (64→128→256→512 channels)
- SE block after each convolution for channel-wise attention
- Global average pooling followed by dense layers for feature recalibration

**U-Net-FusionNet (with Residual Connections):**
- Residual blocks in both encoder and decoder
- Internal skip connections within each block
- Enhanced gradient flow for deeper training

**Learning Rate Selection:**
<p align="left">
  <img src="visuals\test_loss.png" alt="Learning Rate Optimization" width="500">
</p>
  
  - Lowest test loss at 10⁻⁴ for all models

**Convergence Behavior:**
<p align="left">
  <img src="visuals\convergence.png" alt="Convergence Behavior" width="500">
</p>

- Rapid loss drop in first 20 epochs, then plateau
- U-Net-FusionNet: fastest convergence, lowest test loss variability
- U-Net-HRNet: slightly higher early-epoch variance, stabilizes by epoch 25
- All models: minimal improvement after epoch 50

| Architecture | Peak IoU (%) | Dice (%) | Optimal LR | Key Properties |
|---|---|---|---|---|
| Lightweight U-Net | 75.3 | 81.1 | 1e-4 | Efficient baseline; low-resolution features |
| U-Net-HRNet (with SE) | 76.7 | 82.5 | 1e-4 | Channel attention; fine detail preservation |
| U-Net-FusionNet (with Residuals) | 77.1 | 82.8 | 1e-4 | Best boundaries; most stable convergence |
---

### Experiment 3: Data Augmentation Strategy Analysis

- **Methodology:** Offline augmentation with one-by-one application (each original image duplicated per transform)
- **Training Setup:** Batch size 4, Learning rate 1e-4, 50 epochs, threshold 0.5

#### Augmentation Performance by Architecture

**Lightweight U-Net (Shallow Model):**
- No augmentation: 75.3% IoU
- VerticalFlip (best): 78.4% IoU (+3.1)
- Effect: Simple geometric transforms provide robust gains

| Transform | IoU (%) | Effect |
|---|---|---|
| HorizontalFlip | 76.3 | +1.0 |
| VerticalFlip | 78.4 | +3.1 (best) |
| Shift | 75.8 | +0.5 |
| Rotate | 70.5 | -4.8 (harmful) |
| GaussNoise | 71.7 | -3.6 (harmful) |
| Blur | 75.6 | +0.3 |

**U-Net-HRNet (Attention-based):**
- No augmentation: 68.7% IoU
- Blur (best): 80.6% IoU (+11.9)
- Effect: Smoothing operations particularly beneficial; geometric distortions harmful

| Transform | IoU (%) | Effect |
|---|---|---|
| Blur | 80.6 | +11.9 (best) |
| Rotate | 79.9 | +11.2 |
| Shift | 77.1 | +8.4 |
| Scale | 69.2 | +0.5 |
| MotionBlur | 77.0 | +8.3 |
| GaussNoise | 71.1 | +2.4 |

**U-Net-FusionNet (Residual):**
- No augmentation: 77.1% IoU (high baseline)
- VerticalFlip (best): 77.6% IoU (+0.5)
- Effect: Naturally robust; marginal augmentation gains

| Transform | IoU (%) | Effect |
|---|---|---|
| VerticalFlip | 77.6 | +0.5 (best) |
| Rotate | 76.9 | -0.2 |
| GaussNoise | 77.5 | +0.4 |
| Scale | 68.6 | -8.5 (harmful) |
| Blur | 77.2 | +0.1 |

**Critical Findings:**
- No universal strategy across architectures
- Shallow models: geometric flips effective
- Attention models: smoothing operations beneficial
- Residual models: naturally robust, minimal augmentation sensitivity
- Aggressive distortions (heavy rotation, scaling): universally harmful
- Conclusion: Augmentation must be architecture-specific

---

## Key Insights and Contributions

### Multimodal Data Fusion
LiDAR integration provides the most significant performance boost (6.0 IoU points). The gain from structural features (CHM) exceeds spectral band additions. However, using all available data is counterproductive, suggesting selective feature engineering is critical.

### Architecture Sensitivity
Model choice directly determines augmentation strategy effectiveness. FusionNet residual connections provide superior stability and boundary quality. HRNet's attention mechanisms excel with smoothing operations. Lightweight baselines remain efficient references but require careful augmentation tuning.

### Augmentation Strategy
Aggressive transforms universally harm performance, while gentle augmentations (flips, slight shifts) improve robustness. The relationship between model depth/complexity and optimal augmentation intensity is critical for generalization.

### Practical Implications for Forest Monitoring
Results enable rapid, large-scale forest extent mapping at sub-meter resolution using freely available public data (Geobasis Brandenburg). Methodology is reproducible and extensible to other regions with similar geospatial data availability.

---

## Methodology Justification

### Resolution Downsampling (0.2 m to 3.9 m per pixel)

**Why 256×256 pixels (3.9 m/pixel)?**

Original data (5000×5000 px at 0.2 m/pixel) requires:
- 25 million pixels per tile
- 125+ million floats per tile with 5+ channels
- Exceeds practical GPU memory constraints even with 12GB VRAM
- Batch size limited to 1-2 samples, severely hindering training efficiency

**Is this resolution adequate?**

Semantic preservation analysis:
- Single tree crown (10-30 m): 2.6-7.7 pixels at 3.9 m (borderline for individual trees)
- Forest patch/habitat unit (100-500 m): 25.6-128 pixels (excellent)
- Canopy density patterns (30-100 m): 7.7-25.6 pixels (adequate)

This resolution preserves forest-pattern-level information required for binary forest/non-forest classification, the task objective. Tree-level segmentation would require 512×512 patches at 1 m/pixel and is reserved for future work.

**Comparison with literature:**
Standard remote sensing workflows typically use 2-5 m/pixel for forest mapping (Sentinel-2 at 10m, Landsat at 30m)((Bourgoin, 2026; Pilaš et al., 2020)). This study operates at the high-resolution end of this spectrum, sacrificing tree delineation detail to gain computational efficiency while preserving landscape-level forest structure.

---

## Limitations

- **Temporal offset:** 455-day gap between RGBI (April 2024) and LiDAR (January 2023) data. Acceptable for stable canopy; seasonal analysis requires co-temporal acquisition.
- **Single study area:** Results specific to Brandenburg temperate forest. Generalization to other biomes, climates, or forest types requires multi-region validation.
- **Binary classification:** Forest/non-forest only. Multi-class forest type, species, or condition assessment requires different ground-truth and architecture design.
- **Ground-truth artifacts:** SAM-assisted masks required manual GIMP refinement. Fully automated generation could improve scalability.
- **Class imbalance:** Forest coverage varies 30-53% across tiles, requiring stratification and loss weighting for balanced learning.

---

## Future Research Directions

- **Temporal consistency:** Acquire co-seasonal LiDAR+RGBI to eliminate phenological bias and improve seasonal segmentation
- **Transformer architectures:** Evaluate Vision Transformers (ViTs) and hybrid CNN-Transformer designs for improved long-range spatial modeling
- **Uncertainty quantification:** Implement Bayesian layers and Monte Carlo dropout for confidence estimation
- **Hierarchical segmentation:** Forest/non-forest at 256×256 followed by local high-resolution refinement for tree-level delineation
- **Multi-region evaluation:** Train on diverse ecosystems (boreal, temperate, tropical) to assess cross-biome robustness
- **Adaptive augmentation:** Dynamically tailor transforms per-sample to preserve critical forest structures

---

## Installation and Reproducibility

**Requirements:**
- Python 3.10+
- TensorFlow 2.10+
- GeoPandas, Rasterio (geospatial data handling)
- NumPy, Pandas, Scikit-learn, OpenCV

**Setup:**
```bash
pip install -r requirements.txt
jupyter notebook notebooks/Experiments.ipynb
```

**Experiment Reproduction:**
All experiments and results are contained in `notebooks/Experiments.ipynb`. This notebook includes:
- Band-combination experiments
- Model comparison training and validation
- Augmentation sensitivity analysis
- Metric computation (IoU, Dice, Precision, Recall)
- Probability map visualization and interpretation

**Note on Results:**
Experiments documented in this repository were re-run after thesis submission as part of ongoing reproducibility validation. Minor metric variations from the thesis defense presentation reflect code refinement, hyperparameter tuning, and random seed effects. Core findings remain consistent.

---

## Data Access

Public data sources:
- **Orthophotos:** Geobasis Brandenburg (https://geobasis-bb.de)
- **LiDAR:** Geobasis Brandenburg / GeoPortal Brandenburg
- **License:** Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0)

Processed dataset files are not included in this repository due to size constraints. Preprocessing instructions are in `notebooks/preprocess_data.ipynb`.

References
Bourgoin, C. (2026). GFC2020: A global map of forest land use for year 2020 to support the EU Deforestation Regulation. Earth System Science Data, 18, 1331.

Pilaš, I., Gašparović, M., Novkinić, A., & Klobučar, D. (2020). Mapping of the canopy openings in mixed beech–fir forest at Sentinel-2 subpixel level using UAV and machine learning approach. Remote Sensing, 12(23), 3925. https://doi.org/10.3390/rs12233925