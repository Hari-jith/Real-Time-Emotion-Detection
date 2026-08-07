# Real-Time Emotion Detection

A deep learning project for recognizing human facial emotions from images and real-time video streams using transfer learning.

> **Project Status:** 🚧 Work in Progress

The current implementation focuses on training an emotion classification model using **MobileNetV2** on the **RAF-DB** dataset. Future updates will integrate real-time webcam inference, YOLO-based face detection, and model comparison using EfficientNetB0.

---

## Project Objectives

- Train a deep learning model for facial emotion recognition.
- Compare multiple transfer learning architectures.
- Build a real-time webcam-based emotion detection system.
- Detect faces using YOLO.
- Display emotion predictions with confidence scores in real time.

---

## Dataset

**RAF-DB (Real-world Affective Faces Database)**

The dataset contains images labelled into seven basic facial expressions:

| Label | Emotion |
|-------|----------|
| 1 | Surprise |
| 2 | Fear |
| 3 | Disgust |
| 4 | Happy |
| 5 | Sad |
| 6 | Angry |
| 7 | Neutral |

---

## Current Implementation

### Data preprocessing

- Image resizing (224 × 224)
- Normalization
- Data augmentation
  - Rotation
  - Horizontal flip
  - Zoom
  - Width/Height shifting

---

### Model

- MobileNetV2 (ImageNet pretrained)
- Transfer Learning
- Global Average Pooling
- Batch Normalization
- Dense Layer
- Dropout
- Softmax classifier (7 classes)

---

### Training Strategy

#### Phase 1

- Frozen backbone
- Feature extraction

#### Phase 2

- Fine tuning
- Last 20 layers unfrozen
- Low learning rate (1e-5)

---

## Current Results

### MobileNetV2 (Base Model)

| Metric | Value |
|--------|-------|
| Accuracy | **61%** |

Classification Report

| Metric | Score |
|---------|-------|
| Precision (Weighted) | 0.58 |
| Recall (Weighted) | 0.61 |
| F1 Score (Weighted) | 0.58 |

---

## Visualizations

### Training History

*(Insert training accuracy/loss graph here)*

### Fine-Tuning History

*(Insert fine-tuning accuracy/loss graph here)*

### Confusion Matrix

*(Insert confusion matrix here)*

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## Future Work

- [x] MobileNetV2 implementation
- [x] Transfer learning
- [x] Fine tuning
- [ ] EfficientNetB0 implementation
- [ ] Compare MobileNetV2 vs EfficientNetB0
- [ ] Save best performing model
- [ ] YOLO face detection
- [ ] Real-time webcam inference
- [ ] Emotion confidence visualization
- [ ] Temporal smoothing
- [ ] Multiple face detection
- [ ] Streamlit web application
- [ ] Model deployment

---

## Future Pipeline

```
Webcam
    │
    ▼
YOLO Face Detection
    │
    ▼
Face Cropping
    │
    ▼
Emotion Classification Model
    │
    ▼
Emotion Prediction
    │
    ▼
Real-Time Visualization
```
