# 🛰️ Object Detection in Optical Remote Sensing Images  
**Faster R-CNN with Transfer Learning, Data Augmentation, and Performance Analysis**

This repository contains a **notebook-based implementation** for multi-class object detection in optical remote sensing images.  
The project focuses on training and evaluating a **Faster R-CNN (ResNet-50 FPN)** model using **transfer learning**, **data augmentation**, and **detailed performance analysis**.



## Overview

Object detection in remote sensing imagery presents several challenges:
- complex and cluttered backgrounds,
- large variation in object scale,
- small and densely packed objects,
- limited labeled datasets.

This notebook explores a practical deep-learning pipeline for addressing these challenges using **TorchVision’s object detection framework**, with a strong emphasis on **evaluation metrics and model behavior analysis**.

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
  - Pre-trained on ImageNet
- Custom classification head using `FastRCNNPredictor`
- Training setup:
  - Optimizer: **SGD (momentum + weight decay)**
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
  - detection results
- Comparison of model performance under different configurations

### 5) Model Saving
- Trained model serialized using `pickle` (`.pkl`)

---

## Dataset

- Optical remote sensing image dataset
- Multi-class object detection task
- Dataset split:
  - **70% training / 30% testing**


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
  

---

## Repository Structure

