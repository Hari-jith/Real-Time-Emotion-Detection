# Real-Time Video-Based Facial Emotion Detection

A deep learning pipeline for **real-time facial emotion detection from images, videos, and webcam streams**.

The system combines two independent computer vision stages:

1. **YOLO face detection** — detects and localizes human faces in an image or video frame.
2. **CNN-based emotion classification** — classifies each detected face into one of eight facial emotion categories.

The face-detection component uses a YOLO model trained on WIDER FACE, while the emotion-classification component has been rebuilt using the **AffectNet dataset** and evaluated using three transfer-learning CNN architectures:

- MobileNetV2
- DenseNet121
- ResNet50

The three models are trained independently, evaluated on the held-out test set, and compared using multiple classification metrics. The best-performing model is then fine-tuned separately. Based on the current evaluation, **fine-tuned ResNet50 is the final emotion classification model**.

---

## 1. Project Overview

The complete inference pipeline is:

```text
Input Image / Video / Webcam
           |
           v
     YOLO Face Detector
           |
           v
    Face Bounding Boxes
           |
           v
       Face Cropping
           |
           v
     Image Preprocessing
           |
           v
   ResNet50 Emotion Classifier
           |
           v
    Emotion + Confidence
           |
           v
      Annotated Output
````

For every detected face, the system:

* detects the face using YOLO,
* extracts the corresponding face region,
* resizes the cropped face,
* passes it through the trained emotion classifier,
* predicts one of eight emotion classes,
* calculates the prediction confidence,
* displays the face bounding box and predicted emotion.

The same overall pipeline can be used with:

* individual images,
* prerecorded videos,
* real-time webcam streams.

---

# 2. Key Features

* Real-time webcam-based facial emotion detection
* YOLO-based face detection
* AffectNet-based emotion classification
* Eight-class facial emotion recognition
* Transfer learning using pretrained CNN architectures
* Independent training of MobileNetV2, DenseNet121 and ResNet50
* Validation-based comparison of CNN architectures
* Class-weighted training for handling class imbalance
* Slight image augmentation during training
* Fine-tuning only the best-performing CNN
* Classification report with precision, recall and F1-score
* Confusion matrix analysis
* Multiclass ROC-AUC analysis
* Per-class ROC-AUC evaluation
* Model comparison
* Random test-image predictions with confidence scores
* Threaded webcam and inference pipeline
* Real-time FPS monitoring
* Modular separation between face detection and emotion classification

---

# 3. Emotion Classes

The emotion-classification component uses eight emotion categories from the AffectNet dataset.

| Class Index | Emotion  |
| ----------: | -------- |
|           0 | Anger    |
|           1 | Contempt |
|           2 | Disgust  |
|           3 | Fear     |
|           4 | Happy    |
|           5 | Neutral  |
|           6 | Sad      |
|           7 | Surprise |

The CNN classifier uses the following fixed class order during training, evaluation and inference:

```python
CLASS_NAMES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]
```

Keeping this ordering identical during training, evaluation and inference is essential because the output index of the softmax layer corresponds directly to this class mapping.

---

# 4. Dataset

## AffectNet

The emotion-classification stage uses an AffectNet-derived dataset organized into class-specific folders.

The CNN training pipeline reads labels directly from the folder names and does not use `labels.csv`.

```text
Dataset/
├── Train/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
└── Test/
    ├── Anger/
    ├── Contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Folder-name capitalization is normalized during dataset loading.

The `labels.csv` file present with the dataset is explicitly ignored.

### Dataset Distribution

The original training and test distributions are:

### Training Set

| Emotion   |     Images |
| --------- | ---------: |
| Anger     |      1,500 |
| Contempt  |      1,559 |
| Disgust   |      1,229 |
| Fear      |      1,512 |
| Happy     |      2,340 |
| Neutral   |      2,758 |
| Sad       |      3,091 |
| Surprise  |      2,119 |
| **Total** | **16,108** |

### Test Set

| Emotion   |     Images |
| --------- | ---------: |
| Anger     |      1,718 |
| Contempt  |      1,312 |
| Disgust   |      1,248 |
| Fear      |      1,664 |
| Happy     |      2,704 |
| Neutral   |      2,368 |
| Sad       |      1,584 |
| Surprise  |      1,920 |
| **Total** | **14,518** |

The original training set is split into training and validation subsets using a **stratified split**, while the original test set is kept separate for final evaluation.

```text
Original Training Data
        |
        +------------------+
        |                  |
        v                  v
    Training            Validation
      85%                   15%
        |
        |
        v
Original Test Data
        |
        v
Final Independent Test
```

The test set is not used for model selection or training.

---

# 5. Data Preprocessing

Images are resized to:

```text
224 × 224
```

The CNN models use three-channel input suitable for ImageNet-pretrained architectures.

The preprocessing pipeline is:

```text
Input Image
     |
     v
Image Loading
     |
     v
Resize to 224 × 224
     |
     v
Float32 Conversion
     |
     v
Training Augmentation
     |
     v
Model-Specific Preprocessing
     |
     v
CNN Backbone
```

Light augmentation is applied during training to improve generalization.

The augmentation pipeline uses controlled transformations such as:

* horizontal flipping,
* slight rotation,
* slight zoom,
* slight contrast variation.

Augmentation is applied only during model training and is disabled during evaluation and inference.

---

# 6. Class Imbalance Handling

The AffectNet training data is not perfectly balanced.

Class weights were calculated from the training distribution and supplied to the CNN training process.

The calculated weights are:

| Emotion  | Class Weight |
| -------- | -----------: |
| Anger    |        1.342 |
| Contempt |        1.292 |
| Disgust  |        1.638 |
| Fear     |        1.332 |
| Happy    |        0.860 |
| Neutral  |        0.730 |
| Sad      |        0.651 |
| Surprise |        0.950 |

Higher weights are assigned to classes with fewer training samples, allowing the loss function to penalize mistakes on those classes more strongly.

This helps prevent the classifier from being dominated by the more frequently represented emotions.

---

# 7. Face Detection Component

The face-detection stage remains separate from the emotion-classification stage.

## Dataset

The YOLO face detector was trained using **WIDER FACE**.

The WIDER FACE annotations were converted into YOLO-compatible format for training.

The detector uses a single object class:

```text
0 → face
```

The original YOLO workflow processed approximately **12,880 training images**.

The purpose of the YOLO model is strictly face localization.

It does **not** classify emotions.

---

# 8. YOLO Face Detection Workflow

For each input frame:

```text
Input Frame
     |
     v
YOLO Face Detector
     |
     +---- Face 1 Bounding Box
     |
     +---- Face 2 Bounding Box
     |
     +---- Face 3 Bounding Box
              |
              v
        Crop Each Face
```

The detected face regions are then passed to the CNN emotion-classification model.

The trained YOLO model is:

```text
yolo_face_detector_best.pt
```

The YOLO model therefore acts as the front-end detector while the CNN performs emotion classification.

---

# 9. CNN Emotion Classification

Three CNN architectures were trained and evaluated independently:

* MobileNetV2
* DenseNet121
* ResNet50

All three models use ImageNet-pretrained backbones and a common emotion-classification framework.

The models are initially trained with their pretrained backbones frozen.

After all three models have been evaluated, the best-performing architecture is selected.

Only that selected model is then fine-tuned.

```text
AffectNet
    |
    v
Training / Validation / Test
    |
    +------------------+------------------+
    |                  |                  |
    v                  v                  v
MobileNetV2        DenseNet121        ResNet50
    |                  |                  |
    v                  v                  v
Evaluation          Evaluation        Evaluation
    |                  |                  |
    +------------------+------------------+
                       |
                       v
                Model Comparison
                       |
                       v
                 Best Base Model
                       |
                       v
                  Fine-Tuning
                       |
                       v
                Final Evaluation
                       |
                       v
                Final CNN Model
```

---

# 10. CNN Architecture

The three CNN models follow the same general classification-head design.

```text
Input Image
     |
     v
224 × 224 × 3
     |
     v
Training Augmentation
     |
     v
Model-Specific Preprocessing
     |
     v
ImageNet Pretrained CNN Backbone
     |
     v
Global Average Pooling
     |
     v
Batch Normalization
     |
     v
Dropout
     |
     v
Dense Layer - 256 Units
     |
     v
Batch Normalization
     |
     v
Dropout
     |
     v
Softmax Output
     |
     v
8 Emotion Classes
```

---

# 11. MobileNetV2

MobileNetV2 is a lightweight convolutional architecture designed for computational efficiency.

Its main advantages include:

* lower computational requirements,
* relatively fast inference,
* smaller architecture compared with heavier CNNs,
* suitability for real-time applications.

MobileNetV2 was used as the lightweight baseline in this project.

### Test Performance

```text
Test Loss:       1.3952
Test Accuracy:   49.32%
Macro Precision: 46.15%
Macro Recall:    46.30%
Macro F1:        45.61%
Weighted F1:     48.52%
Macro ROC-AUC:   0.8461
```

---

# 12. DenseNet121

DenseNet121 uses dense connections between convolutional layers, allowing feature information to be reused throughout the network.

Its characteristics include:

* feature reuse,
* efficient gradient propagation,
* strong feature representation,
* relatively deep architecture.

### Test Performance

```text
Test Loss:       1.4617
Test Accuracy:   52.04%
Macro Precision: 49.25%
Macro Recall:    49.04%
Macro F1:        47.66%
Weighted F1:     50.94%
Macro ROC-AUC:   0.8598
```

DenseNet121 performed better than MobileNetV2 on the overall classification metrics.

---

# 13. ResNet50

ResNet50 uses residual connections to make deeper neural networks easier to optimize.

Its characteristics include:

* residual learning,
* strong feature extraction,
* deep feature representation,
* effective transfer-learning performance.

Among the three base models, ResNet50 produced the strongest overall results.

### Base ResNet50 Test Performance

```text
Test Loss:       1.3857
Test Accuracy:   53.80%
Macro Precision: 50.20%
Macro Recall:    50.30%
Macro F1:        49.73%
Weighted F1:     53.07%
Macro ROC-AUC:   0.8657
```

ResNet50 was therefore selected as the best base model for further fine-tuning.

---

# 14. Model Comparison

The three CNN architectures were compared using multiple evaluation metrics rather than accuracy alone.

| Model        | Test Accuracy | Macro Precision | Macro Recall |   Macro F1 | Weighted F1 | Macro ROC-AUC |
| ------------ | ------------: | --------------: | -----------: | ---------: | ----------: | ------------: |
| MobileNetV2  |        49.32% |          46.15% |       46.30% |     45.61% |      48.52% |        0.8461 |
| DenseNet121  |        52.04% |          49.25% |       49.04% |     47.66% |      50.94% |        0.8598 |
| **ResNet50** |    **53.80%** |      **50.20%** |   **50.30%** | **49.73%** |  **53.07%** |    **0.8657** |

ResNet50 achieved the strongest overall performance among the three base models.

It was therefore selected for fine-tuning.

---

# 15. Model Training Results

## MobileNetV2

![MobileNetV2 Training Curves](results/CNN/MobileNetV2_training_curves.png)

The training and validation curves are used to monitor:

* convergence,
* training progress,
* potential overfitting,
* potential underfitting,
* differences between training and validation performance.

---

## DenseNet121

![DenseNet121 Training Curves](results/CNN/DenseNet121_training_curves.png)

The same training and validation metrics are monitored for DenseNet121.

---

## ResNet50

![ResNet50 Training Curves](results/CNN/ResNet50_training_curves.png)

ResNet50 was selected for further fine-tuning after comparing the independent baseline results.

---

# 16. Confusion Matrix Analysis

Confusion matrices are generated separately for all three baseline models.

They provide a class-level view of the model's predictions and show which emotion categories are frequently confused with one another.

## MobileNetV2

![MobileNetV2 Confusion Matrix](results/CNN/MobileNetV2_confusion_matrix.png)

## DenseNet121

![DenseNet121 Confusion Matrix](results/CNN/DenseNet121_confusion_matrix.png)

## ResNet50

![ResNet50 Confusion Matrix](results/CNN/ResNet50_confusion_matrix.png)

The confusion matrices are particularly useful for identifying classes that are difficult to distinguish based only on overall accuracy.

---

# 17. ROC-AUC Analysis

Multiclass ROC curves are generated using a one-vs-rest approach.

ROC-AUC provides an additional measure of how effectively the classifier separates each emotion from the remaining classes.

## MobileNetV2

![MobileNetV2 ROC-AUC](results/CNN/MobileNetV2_roc_auc.png)

## DenseNet121

![DenseNet121 ROC-AUC](results/CNN/DenseNet121_roc_auc.png)

## ResNet50

![ResNet50 ROC-AUC](results/CNN/ResNet50_roc_auc.png)

---

# 18. ResNet50 Fine-Tuning

After evaluating the three base models, **ResNet50 was selected for fine-tuning**.

The other two models were not fine-tuned because the objective was to improve the strongest candidate while avoiding unnecessary additional training time and computational cost.

The fine-tuning process uses:

```text
Best Base Model
      |
      v
ResNet50
      |
      v
Unfreeze Selected Backbone Layers
      |
      v
Lower Learning Rate
      |
      v
Fine-Tuning
      |
      v
Validation Monitoring
      |
      v
Final Test Evaluation
```

The fine-tuned model is then compared against the original frozen-backbone ResNet50.

---

# 19. Fine-Tuned ResNet50 Performance

The fine-tuned ResNet50 achieved:

```text
Test Loss:       1.2586
Test Accuracy:   62.07%
Macro Precision: 58.90%
Macro Recall:    58.70%
Macro F1:        58.30%
Weighted F1:     61.30%
```

Compared with the original ResNet50:

| Metric          | Base ResNet50 | Fine-Tuned ResNet50 |
| --------------- | ------------: | ------------------: |
| Test Loss       |        1.3857 |          **1.2586** |
| Test Accuracy   |        53.80% |          **62.07%** |
| Macro Precision |        0.5020 |          **0.5890** |
| Macro Recall    |        0.5030 |          **0.5870** |
| Macro F1        |        0.4973 |          **0.5830** |
| Weighted F1     |        0.5307 |          **0.6130** |

Fine-tuning produced a substantial improvement over the frozen-backbone ResNet50.

### Accuracy Improvement

```text
53.80% → 62.07%
```

### Macro F1 Improvement

```text
0.4973 → 0.5830
```

The fine-tuned model is therefore selected as the current final emotion classifier.

---

# 20. Fine-Tuned ResNet50 Classification Report

| Emotion              | Precision |    Recall |  F1-score |
| -------------------- | --------: | --------: | --------: |
| Anger                |     0.582 |     0.430 |     0.495 |
| Contempt             |     0.572 |     0.595 |     0.583 |
| Disgust              |     0.403 |     0.514 |     0.452 |
| Fear                 |     0.596 |     0.419 |     0.492 |
| Happy                |     0.808 |     0.923 |     0.861 |
| Neutral              |     0.732 |     0.830 |     0.778 |
| Sad                  |     0.582 |     0.569 |     0.575 |
| Surprise             |     0.441 |     0.412 |     0.426 |
| **Accuracy**         |           |           | **0.621** |
| **Macro Average**    | **0.589** | **0.587** | **0.583** |
| **Weighted Average** | **0.615** | **0.621** | **0.613** |

The strongest class-level performance is observed for **Happy** and **Neutral**, while **Disgust**, **Fear**, and **Surprise** remain more challenging.

---

# 21. Fine-Tuned ResNet50 ROC-AUC

The per-class ROC-AUC values for the final fine-tuned ResNet50 model are:

| Emotion  | ROC-AUC |
| -------- | ------: |
| Anger    |   0.903 |
| Contempt |   0.897 |
| Disgust  |   0.870 |
| Fear     |   0.902 |
| Happy    |   0.987 |
| Neutral  |   0.965 |
| Sad      |   0.924 |
| Surprise |   0.743 |

The final model demonstrates strong class discrimination for **Happy** and **Neutral**.

The ROC-AUC for **Surprise** is comparatively lower, indicating that this class remains more difficult to separate from the other emotion categories.

---

# 22. Final Model

The final model selected for the complete system is:

```text
Fine-Tuned ResNet50
```

The final pipeline is therefore:

```text
Webcam / Image / Video
          |
          v
   YOLO Face Detector
          |
          v
     Face Cropping
          |
          v
     Fine-Tuned ResNet50
          |
          v
     8-Class Prediction
          |
          v
 Emotion + Confidence
```

### Final Test Performance

| Metric          | Final ResNet50 |
| --------------- | -------------: |
| Test Loss       |     **1.2586** |
| Test Accuracy   |     **62.07%** |
| Macro Precision |      **0.589** |
| Macro Recall    |      **0.587** |
| Macro F1        |      **0.583** |
| Weighted F1     |      **0.613** |

---

# 23. Final Model Random Predictions

The notebook also performs random predictions using images from the held-out test set.

Each prediction displays:

* the input face image,
* actual emotion,
* predicted emotion,
* prediction confidence.

![Final Model Random Predictions](results/CNN/final_random_predictions.png)

These predictions provide a qualitative check of how the final model behaves on individual unseen samples.

---

# 24. Real-Time Video Pipeline

The final video-processing pipeline is:

```text
Input Video
     |
     v
Read Frame
     |
     v
YOLO Face Detection
     |
     v
Crop Detected Faces
     |
     v
Fine-Tuned ResNet50
     |
     v
Predict Emotion
     |
     v
Draw Bounding Boxes
     |
     v
Draw Emotion + Confidence
     |
     v
Output Video
```

The YOLO model remains responsible only for detecting faces.

The fine-tuned ResNet50 model performs the emotion classification.

---

# 25. Real-Time Webcam Detection

A standalone Python script is provided for real-time webcam inference.

The webcam pipeline is:

```text
Webcam
   |
   v
Live Frame
   |
   v
YOLO Face Detection
   |
   v
Face Crop
   |
   v
Fine-Tuned ResNet50
   |
   v
Emotion + Confidence
   |
   v
Live Display
```

The webcam application displays:

* detected face bounding boxes,
* predicted emotion,
* emotion confidence,
* real-time FPS.

The camera and inference operations are separated using a threaded pipeline to prevent model inference from unnecessarily blocking webcam frame acquisition.

---

# 26. CNN Preprocessing During Inference

The final Keras model contains the model-specific preprocessing layer used during training.

Therefore, the webcam application does **not** apply the ResNet50 preprocessing operation a second time.

The inference flow is:

```text
Detected Face
     |
     v
BGR → RGB
     |
     v
Resize to 224 × 224
     |
     v
Float32 Conversion
     |
     v
Fine-Tuned ResNet50
     |
     v
Embedded Preprocessing
     |
     v
Emotion Prediction
```

This prevents double preprocessing during inference.

---

# 27. Project Structure

```text
real-time-emotion-detection/
│
├── notebooks/
│   ├── AffectNet_Emotion_Classification.ipynb
│   └── YOLO_Face_Detection.ipynb
│
├── src/
│   └── webcam_detection.py
│
├── results/
│   ├── CNN/
│   │   ├── MobileNetV2_training_curves.png
│   │   ├── MobileNetV2_confusion_matrix.png
│   │   ├── MobileNetV2_roc_auc.png
│   │   ├── DenseNet121_training_curves.png
│   │   ├── DenseNet121_confusion_matrix.png
│   │   ├── DenseNet121_roc_auc.png
│   │   ├── ResNet50_training_curves.png
│   │   ├── ResNet50_confusion_matrix.png
│   │   ├── ResNet50_roc_auc.png
│   │   └── final_random_predictions.png
│   │
|   ├── integration/
|   |    ├── emotion_detection.png
│   │   ├── emotion_detection_output.mp4
|   |
│   └── demo.gif
│
├── requirements.txt
├── .gitignore
└── README.md
```

Large trained model files and datasets should not be committed directly to the repository unless they are managed through an appropriate large-file mechanism.

---

# 28. Technologies Used

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* Ultralytics YOLO

### Computer Vision

* OpenCV

### Data Processing

* NumPy
* Pandas

### Machine Learning

* Scikit-learn

### Visualization

* Matplotlib
* Seaborn

### Development / Experimentation

* Google Colab
* Jupyter Notebook
* VS Code

---

# 29. Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

For the local webcam application, the system should have:

* a working webcam,
* compatible OpenCV installation,
* TensorFlow,
* Ultralytics,
* the trained emotion-classification model,
* the trained YOLO face-detection model.

---

# 30. Running the CNN Notebook

The CNN notebook is responsible for training and evaluating the emotion-classification models.

The workflow is:

```text
1. Mount Google Drive
2. Load AffectNet dataset
3. Ignore labels.csv
4. Read labels from class folders
5. Analyze class distribution
6. Split training data into training and validation sets
7. Apply preprocessing
8. Apply slight training augmentation
9. Calculate class weights
10. Build MobileNetV2
11. Train MobileNetV2
12. Evaluate MobileNetV2
13. Build DenseNet121
14. Train DenseNet121
15. Evaluate DenseNet121
16. Build ResNet50
17. Train ResNet50
18. Evaluate ResNet50
19. Compare the three base models
20. Select the best model
21. Fine-tune only the selected model
22. Evaluate the fine-tuned model
23. Compare base and fine-tuned performance
24. Save the final model
25. Generate random test predictions
```

The original test set is kept separate and is used for final evaluation.

---

# 31. Running the Webcam Application

After obtaining the final fine-tuned ResNet50 model and keeping the trained YOLO face detector, update the model paths in:

```text
src/webcam_detection.py
```

Example:

```python
CNN_MODEL_PATH = r"C:\path\to\best_emotion_model.keras"
YOLO_MODEL_PATH = r"C:\path\to\yolo_face_detector_best.pt"
```

Then run:

```bash
python src/webcam_detection.py
```

The application opens the webcam and performs:

```text
Face Detection
      ↓
Face Cropping
      ↓
Emotion Classification
      ↓
Confidence Display
```

Press:

```text
Q
```

or:

```text
ESC
```

to stop the application.

---

# 32. Current Best Model

The current final emotion-classification model is:

## Fine-Tuned ResNet50

### Test Performance

**Test Accuracy:** 62.07%

**Macro Precision:** 0.589

**Macro Recall:** 0.587

**Macro F1:** 0.583

**Weighted F1:** 0.613

**Test Loss:** 1.2586

The fine-tuned ResNet50 improves substantially over the original frozen-backbone ResNet50.

```text
Base ResNet50
53.80% Accuracy
       |
       | Fine-Tuning
       v
Fine-Tuned ResNet50
62.07% Accuracy
```

The YOLO face detector remains unchanged.

---

# 33. Important Observations

The final results show that fine-tuning the selected ResNet50 substantially improves emotion classification.

However, performance varies considerably between emotion classes.

The strongest class-level results are observed for:

* Happy
* Neutral
* Sad
* Contempt

More challenging classes include:

* Disgust
* Fear
* Surprise

The final model achieves a particularly strong ROC-AUC for:

```text
Happy    → 0.987
Neutral  → 0.965
Sad      → 0.924
Anger    → 0.903
Fear     → 0.902
Contempt → 0.897
Disgust  → 0.870
Surprise → 0.743
```

This demonstrates why overall accuracy alone is not sufficient for evaluating facial emotion classification.

Macro-level metrics, class-wise F1-scores, confusion matrices and ROC-AUC are considered together.

---

# 34. Limitations

Although the final ResNet50 model provides a substantial improvement over the initial CNN baselines, facial emotion recognition remains a challenging computer vision problem.

The current system can be affected by:

* lighting conditions,
* head pose,
* facial occlusion,
* face size,
* image quality,
* camera quality,
* ambiguous facial expressions,
* similarities between emotion categories,
* differences between dataset images and real-world webcam images.

The lower performance of some emotion categories also indicates that the model does not classify all emotions equally well.

The reported test metrics should therefore not be interpreted as universal real-world emotion recognition accuracy.

Facial expressions are observable visual patterns and should not be treated as definitive measurements of a person's internal emotional state.

---

# 35. Future Improvements

Potential future improvements include:

* improved handling of difficult emotion classes,
* targeted analysis of Disgust, Fear and Surprise,
* face alignment before emotion classification,
* temporal smoothing across consecutive video frames,
* face tracking,
* prediction stabilization,
* confidence thresholding,
* uncertainty estimation,
* improved real-time inference optimization,
* GPU-optimized deployment,
* model quantization,
* ONNX-based deployment,
* evaluation on additional facial-expression datasets,
* improved domain generalization between controlled datasets and real-world webcam images.

---

# 36. Final System Architecture

```text
                         ┌─────────────────────────┐
                         │ Image / Video / Webcam  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    YOLO Face Detector   │
                         │       WIDER FACE        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                              Face Bounding Boxes
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │     Face Cropping       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Image Preprocessing  │
                         │       224 × 224        │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │  Fine-Tuned ResNet50    │
                         │       AffectNet         │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Emotion + Confidence    │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Annotated Live / Video  │
                         │       Output            │
                         └─────────────────────────┘
```

---

# 37. Conclusion

This project combines **YOLO-based face detection** with **deep-learning-based facial emotion classification** to build an end-to-end real-time facial emotion detection system.

The face-detection component uses a YOLO model trained on WIDER FACE and remains unchanged throughout the CNN experimentation.

The emotion-classification component was rebuilt using the AffectNet dataset and evaluated through three independent transfer-learning CNN architectures:

* MobileNetV2
* DenseNet121
* ResNet50

The three models were evaluated using accuracy, precision, recall, F1-score, confusion matrices and ROC-AUC.

ResNet50 achieved the strongest baseline performance and was selected for fine-tuning.

The fine-tuned ResNet50 improved the test accuracy from:

```text
53.80% → 62.07%
```

and the Macro F1-score from:

```text
0.4973 → 0.5830
```

The final system therefore uses:

```text
YOLO Face Detector
        +
Fine-Tuned ResNet50
        +
AffectNet Emotion Classification
        =
Real-Time Facial Emotion Detection
```

The project demonstrates a complete computer-vision workflow covering **dataset preparation, class-imbalance handling, transfer learning, model comparison, selective fine-tuning, multiclass evaluation and real-time inference**.
