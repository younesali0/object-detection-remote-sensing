# 🛰️ Object Detection in Optical Remote Sensing Images  
**Adaptive Mask R-CNN with Hybrid Optimization**

This project explores **multi-class object detection in optical remote sensing images** using deep learning, with a focus on **multi-scale objects, complex backgrounds, and limited labeled data**.

The implementation is based on **Mask R-CNN with a ResNet-50 backbone**, enhanced through **transfer learning, data augmentation, and a hybrid Adam–SGD optimization strategy**.

---

## Overview

Object detection in remote sensing imagery is challenging due to:
- high visual complexity,
- large variation in object scale,
- small and densely packed objects,
- scarcity of annotated data.

This project investigates how an **adaptive training strategy** can improve detection accuracy and generalization under these constraints.

---

## Approach

- **Model**: Mask R-CNN with ResNet-50 (pre-trained on ImageNet)
- **Training strategy**:
  - staged fine-tuning to manage limited GPU memory,
  - transfer learning from natural images,
  - extensive data augmentation (rotation, flipping, translation).
- **Optimization**:
  - comparison of multiple optimizers (Adam, SGD, RMSprop, AdaDelta),
  - adoption of a **hybrid Adam → SGD strategy** for improved generalization.

All experiments and analysis are contained in a single Jupyter notebook.

---

## Dataset

- **NWPU VHR-10 dataset**
- 10 object classes:
  - Airplane, Ship, Storage Tank, Baseball Diamond, Tennis Court,
    Basketball Court, Ground Track Field, Harbor, Bridge, Vehicle
- Train/test split: **70% / 30%**

> Dataset files are not included due to licensing restrictions.

---

## Results

- **Mean Average Precision (mAP): 95%**
- Up to **+6% improvement in AP** compared to baseline detectors:
  - Faster R-CNN
  - YOLO / YOLOv2
  - SSD
  - R-FCN
- Hybrid Adam–SGD optimization showed better generalization than single optimizers.

---

## Technologies Used

- Python  
- PyTorch  
- TensorFlow / Keras  
- Mask R-CNN  
- OpenCV  
- Computer Vision  
- Deep Learning  
- Transfer Learning  

---

