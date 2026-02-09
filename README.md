# 🛰️ Object Detection in Optical Remote Sensing Images  
**Faster R-CNN with Transfer Learning, Data Augmentation, and Performance Analysis**

This repository contains a **notebook-based implementation** for multi-class object detection in optical remote sensing images.  
The project focuses on training and evaluating a **Faster R-CNN (ResNet-50 FPN)** model using **transfer learning**, **data augmentation**, and **detailed performance analysis**.

> 📌 Repository format: **one Jupyter notebook + README**  


---

## Overview

Object detection in remote sensing imagery presents several challenges:
- complex and cluttered backgrounds,
- large variation in object scale,
- small and densely packed objects,
- limited labeled datasets.

This project explores a practical deep-learning pipeline for addressing these challenges using **TorchVision’s object detection framework**, with a strong emphasis on **evaluation metrics and model behavior analysis**.

---

## What’s Inside the Notebook

### 1) Data Preparation
- Loading optical remote sensing images
- Parsing object annotations (bounding boxes + labels)
- Handling positive and negative samples
- Image resizing while preserving bounding box consistency

### 2) Data Augmentation
- Implemented using **Albumentations**
- Includes:
  - horizontal flipping
- Bounding boxes handled in **Pascal VOC format**

### 3) Model & Training
- **Model**: Faster R-CNN with ResNet-50 FPN  
  (pre-trained on ImageNet)
- Custom classification head using `FastRCNNPredictor`
- Training configuration:
  - Optimizer: **SGD** (momentum + weight decay)
  - Learning rate scheduler: **StepLR**
- Training and evaluation rely on **TorchVision reference utilities**:
  - `train_one_epoch`
  - `evaluate`
  - custom `collate_fn`

### 4) Evaluation & Analysis
- Quantitative evaluation using:
  - **Average Precision (AP)**
  - **Intersection over Union (IoU)**
  - **Precision–Recall analysis**
- Visualization of:
  - training behavior
  - detection outputs
- Comparison of model performance under different configurations

### 5) Model Saving
- Trained model serialized using `pickle` (`.pkl`)

---

## Dataset

The experiments in this project are conducted using the **NWPU VHR-10 (Very High Resolution)** remote sensing dataset.

**Dataset characteristics:**
- **Source**: Cropped from Google Earth imagery
- **Annotation**: Manually labeled by domain experts
- **Image type**: Optical remote sensing images
- **Task**: Multi-class object detection using bounding boxes

**Object categories (10 classes):**
- Airplane  
- Ship  
- Storage Tank  
- Baseball Diamond  
- Tennis Court  
- Basketball Court  
- Ground Track Field  
- Harbor  
- Bridge  
- Vehicle  

**Data split:**
- **70% training**
- **30% testing**

**Key challenges addressed by the dataset:**
- Large variation in object scales
- Small and densely packed objects
- Complex and cluttered backgrounds
- Class imbalance across categories

The NWPU VHR-10 dataset is commonly used to evaluate object detection models under realistic remote sensing conditions, using metrics such as **Average Precision (AP)**, **Intersection over Union (IoU)**, and **Precision–Recall Curves (PRC)**.



---

## Technologies & Libraries

**Core**
- Python
- PyTorch
- TorchVision (Faster R-CNN Detection API)

**Data & Augmentation**
- Albumentations (+ `ToTensorV2`)
- OpenCV (cv2)
- Pillow (PIL)

**Scientific Stack**
- NumPy
- Pandas
- Matplotlib

**Utilities**
- pycocotools (used by TorchVision evaluation tools)
- XML parsing (`xml.etree`) for annotation handling

**Environment**
- Developed primarily in **Google Colab**  
  (can be adapted to local execution)

---

## Repository Structure

.
├── object_detection_faster_rcnn.ipynb # Main notebook (training + evaluation)
└── README.md


---

## How to Run

1. Clone the repository
2. Open the notebook:
   ```bash
   jupyter notebook object_detection_faster_rcnn.ipynb
Update dataset paths in the configuration cells

Run the notebook cells sequentially

Some cells related to Google Drive mounting are optional and can be skipped for local execution.

Configuration Notes
Dataset paths are defined directly in the notebook

Absolute paths (local / Colab) should be updated before execution

No credentials or private data are included

Notes & Limitations
This project is academic and research-oriented

Not intended as a production-ready pipeline

Focuses on model behavior, evaluation, and experimentation

Code is provided for educational and demonstration purposes

References
A. S. Mahmoud, A. A. Abdelwahab, and M. A. Elattar,
“Object Detection Using Adaptive Mask R-CNN in Optical Remote Sensing Images,”
International Journal of Intelligent Engineering and Systems,
vol. 13, no. 1, pp. 24–35, 2020.

