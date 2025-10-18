# Evaluating the Impact of Data Augmentation on Forest Segmentation Using Aerial Imagery

**Author:** Aylin Gülüm  
**Institution:** Hochschule für Technik und Wirtschaft Berlin (HTW Berlin)  
**Program:** M.Sc. Project Management & Data Science  
**Thesis Duration:** October 2024 – February 2025  
**Supervisor:** Prof. Stefan Wittenberg  
**Second Supervisor:** Prof. Haack  

---

## Abstract

This thesis investigates the impact of data augmentation and multimodal data fusion on forest segmentation performance using aerial imagery and LiDAR data.  
Through the integration of RGBI orthophotos and LiDAR-derived canopy height models (CHM), the study explores how combining spectral and structural features affects model accuracy, robustness, and generalization.  
Three architectures were compared — **U-Net**, **U-Net-HRNet**, and **U-Net-FusionNet** — under various augmentation strategies to assess performance on high-resolution forest segmentation tasks.

Results show that integrating LiDAR-derived CHM with RGBI imagery improves segmentation accuracy by 5–7%, with **U-Net-FusionNet** achieving the best results (IoU: 0.88, Dice: 0.92).  
The study highlights the significance of data fusion and augmentation design in enhancing segmentation for environmental monitoring and sustainable forestry.

---

## Repository Structure

masters_docs/
- ├── notebooks          # Preprocessing, visualization, and exploratory notebooks
- ├── models           # U-Net, HRNet, and FusionNet model architectures
- ├── training         # Scripts for model training and evaluation
- ├── results         # Visual results, metrics, and output masks
- ├── utils           # Helper scripts for augmentation, I/O, and patch extraction
- ├── requirements.txt    # Python dependencies
- └── README.md           # Project documentation

---

## Methodology

1. **Data Preprocessing**
   - Alignment of RGBI and LiDAR tiles  
   - CHM generation from LiDAR point clouds  
   - Patch extraction and normalization  
   - Forest mask binarization for segmentation tasks  

2. **Model Architectures**
   - **U-Net:** Baseline convolutional segmentation model  
   - **U-Net-HRNet:** Enhanced encoder with high-resolution feature retention  
   - **U-Net-FusionNet:** Late-fusion network combining spectral and structural inputs  

3. **Augmentation Techniques**
   - Geometric (rotation, flips, scaling)  
   - Photometric (brightness and contrast shifts)  
   - Noise-based (Gaussian and salt-and-pepper noise)  
   - Spectral dropout for multispectral data  

4. **Evaluation Metrics**
   - Intersection over Union (IoU)  
   - Dice Coefficient  
   - Precision and Recall  

---

## Dataset

**Region:** Tschernitz, Brandenburg (Germany)  
**Data Source:** [Landesvermessung und Geobasisinformation Brandenburg (LGB)](https://geobasis-bb.de)  
**Portal:** [GeoPortal Brandenburg – Open Data](https://geoportal.brandenburg.de)  
**License:** *Datenlizenz Deutschland – Namensnennung – Version 2.0 (dl-de/by-2-0)*  

**Data Description:**
- Aerial orthophotos (RGBI, 20–30 cm resolution)  
- LiDAR-derived CHM (~1 m resolution)  
- Ground-truth forest masks from official land cover datasets  

> Note: Raw data cannot be redistributed here due to license restrictions.  
> Users may download identical datasets directly from the LGB Open Data Portal.

---

## Installation and Reproduction

This project was developed with **Python 3.10** and **PyTorch 2.2**.  
Clone the repository and install dependencies as follows: