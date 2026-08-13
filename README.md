# Real-Time Video-Based Emotion Detection

A deep learning pipeline for **real-time human emotion detection from
images, videos, and webcam streams**.

The system combines two independent computer vision stages:

1.  **YOLO face detection** --- detects and localizes faces in an
    image/video frame.
2.  **CNN-based emotion classification** --- classifies each detected
    face into one of seven basic emotions.

The YOLO face-detection component is retained from the previous version
of the project. The CNN emotion-classification stage has been rebuilt
and evaluated using **MobileNetV2, DenseNet121, and ResNet50**, with
**ResNet50 currently selected as the best-performing classifier** based
on the reported evaluation results.

------------------------------------------------------------------------

## 1. Project Overview

The complete inference pipeline is:

``` text
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
   CNN Preprocessing
          |
          v
   Emotion Classifier
          |
          v
 Emotion + Confidence
          |
          v
Annotated Output
```

For every detected face, the system:

-   detects the face using YOLO,
-   extracts the corresponding face region,
-   preprocesses the cropped face,
-   passes it through the trained CNN classifier,
-   predicts one of seven emotions,
-   calculates the prediction confidence,
-   draws the face bounding box and emotion label on the output.

The same pipeline can be used for:

-   individual images,
-   prerecorded videos,
-   real-time webcam streams.

------------------------------------------------------------------------

# 2. Key Features

-   Face detection using a trained YOLO model.
-   Seven-class facial emotion classification.
-   RAF-DB based emotion classifier.
-   Transfer-learning based CNN models.
-   Comparison of MobileNetV2, DenseNet121 and ResNet50.
-   Training and validation accuracy/loss monitoring.
-   Confusion matrix evaluation.
-   Classification report with precision, recall and F1-score.
-   Multiclass ROC-AUC analysis.
-   Per-class ROC-AUC scores.
-   Model performance comparison.
-   Approximate inference-speed comparison.
-   Prerecorded video emotion detection.
-   Real-time webcam emotion detection.
-   Modular separation between face detection and emotion
    classification.

------------------------------------------------------------------------

# 3. Emotion Classes

The project uses the seven basic emotion classes provided by the RAF-DB
setup.

    Label Emotion
  ------- ----------
        1 Surprise
        2 Fear
        3 Disgust
        4 Happy
        5 Sad
        6 Angry
        7 Neutral

The CNN classifier uses the following fixed class order during
prediction:

``` python
CLASS_NAMES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]
```

Keeping this ordering identical during training, evaluation and
inference is essential. The same mapping must also be used by the final
webcam integration.

------------------------------------------------------------------------

# 4. Dataset

## RAF-DB

The emotion-classification stage uses the **RAF-DB dataset**.

The dataset is organized using class folders rather than requiring the
CSV label files during the CNN training pipeline.

``` text
RAF-DB DATASET/
└── DATASET/
    ├── train/
    │   ├── 1/
    │   │   ├── train_....jpg
    │   │   └── ...
    │   ├── 2/
    │   ├── 3/
    │   ├── 4/
    │   ├── 5/
    │   ├── 6/
    │   └── 7/
    │
    └── test/
        ├── 1/
        │   ├── test_0002_aligned.jpg
        │   ├── test_0004_aligned.jpg
        │   ├── test_0008_aligned.jpg
        │   └── ...
        ├── 2/
        ├── 3/
        ├── 4/
        ├── 5/
        ├── 6/
        └── 7/
```

The numeric folder names correspond directly to the emotion mapping
defined above.

The CNN notebook therefore reads the images from the class directories
and does not depend on loading the RAF-DB CSV files for the image
labels.

------------------------------------------------------------------------

# 5. Face Detection Component

The face-detection stage remains unchanged from the previous version of
the project.

## Dataset

The YOLO face detector was trained using **WIDER FACE**.

The WIDER FACE data was converted into YOLO-compatible annotation format
for training.

The detector uses a single object class:

``` text
0 -> face
```

The YOLO dataset preparation and training stage processed the WIDER FACE
training data, with approximately **12,880 training images** represented
in the original dataset workflow.

The purpose of this model is strictly face localization. It does not
classify emotions.

------------------------------------------------------------------------

# 6. YOLO Face Detection Workflow

For each input frame:

``` text
Frame
  |
  v
YOLO Face Detector
  |
  +---- Face 1 bounding box
  |
  +---- Face 2 bounding box
  |
  +---- Face 3 bounding box
             |
             v
       Crop each face
```

The detected bounding boxes are then passed to the CNN
emotion-classification stage.

The YOLO model remains the same as in the previous implementation and is
loaded from:

``` text
yolo_face_detector_best.pt
```

The YOLO model therefore acts as the front-end detector while the CNN
acts as the emotion-classification model.

------------------------------------------------------------------------

# 7. CNN Emotion Classification

The previous CNN implementation was rebuilt to compare three
transfer-learning architectures:

-   MobileNetV2
-   DenseNet121
-   ResNet50

The objective was not simply to compare architectures, but to identify a
classifier that provides a better balance between:

-   classification accuracy,
-   macro-level performance,
-   per-class performance,
-   ROC-AUC,
-   inference speed.

------------------------------------------------------------------------

# 8. CNN Models

## MobileNetV2

MobileNetV2 is a lightweight CNN architecture designed for computational
efficiency.

Advantages:

-   relatively low computational cost,
-   fast inference,
-   suitable for real-time applications,
-   smaller model compared with heavier architectures.

However, in the current evaluation it produced the lowest classification
performance among the three models.

------------------------------------------------------------------------

## DenseNet121

DenseNet121 uses dense connections between convolutional layers.

Advantages:

-   feature reuse,
-   efficient gradient propagation,
-   strong representation learning,
-   generally good performance on image-classification tasks.

DenseNet121 improved substantially over MobileNetV2 in the current
evaluation.

------------------------------------------------------------------------

## ResNet50

ResNet50 uses residual connections to make deeper networks easier to
optimize.

Advantages:

-   strong feature extraction,
-   effective deep representation learning,
-   robust transfer-learning performance,
-   strong performance on the RAF-DB classification task used in this
    project.

Based on the current test results, **ResNet50 is the best-performing CNN
model** and is the current candidate for the final real-time
integration.

------------------------------------------------------------------------

# 9. Training Pipeline

The CNN notebook follows a transfer-learning based workflow:

``` text
RAF-DB
   |
   v
Folder-based Dataset Loading
   |
   v
Image Preprocessing
   |
   v
Training Augmentation
   |
   v
Class Imbalance Handling
   |
   v
Pretrained CNN Backbone
   |
   v
Emotion Classification Head
   |
   v
Initial Training
   |
   v
Fine-Tuning
   |
   v
Validation Monitoring
   |
   v
Best Model Selection
   |
   v
Independent Test Evaluation
```

The dataset preparation includes image augmentation and class-imbalance
handling so that the models do not simply optimize for the dominant
emotion classes.

The final model is evaluated on the independent test set.

------------------------------------------------------------------------

# 10. Evaluation Methodology

Each model was evaluated using multiple complementary metrics.

### Classification Metrics

-   Accuracy
-   Macro Precision
-   Macro Recall
-   Macro F1-score
-   Weighted F1-score

### Probabilistic Evaluation

-   Multiclass ROC-AUC
-   Per-class ROC-AUC

### Diagnostic Visualizations

-   Training accuracy
-   Validation accuracy
-   Training loss
-   Validation loss
-   Confusion matrix
-   Per-class ROC curves
-   Model comparison chart

Inference timing was also measured to understand the practical
difference between the models.

------------------------------------------------------------------------

# 11. Final CNN Results

The reported independent test-set results are:

  ---------------------------------------------------------------------------------------------------------
  Model             Test Loss     Accuracy        Macro Macro Recall     Macro F1  Weighted F1        Macro
                                              Precision                                             ROC-AUC
  -------------- ------------ ------------ ------------ ------------ ------------ ------------ ------------
  MobileNetV2          1.2029       0.5655       0.4977       0.5756       0.5105       0.5860       0.8809

  DenseNet121          1.0314       0.6229       0.5273       0.5997       0.5442       0.6442       0.9053

  **ResNet50**     **0.9183**   **0.6705**   **0.5659**   **0.6417**   **0.5886**   **0.6853**   **0.9184**
  ---------------------------------------------------------------------------------------------------------

ResNet50 is the strongest model across the primary overall metrics in
the current evaluation.

Compared with MobileNetV2, ResNet50 improves:

-   accuracy from **56.55% → 67.05%**
-   macro F1 from **0.5105 → 0.5886**
-   weighted F1 from **0.5860 → 0.6853**
-   macro ROC-AUC from **0.8809 → 0.9184**

DenseNet121 provides an intermediate result between MobileNetV2 and
ResNet50.

------------------------------------------------------------------------

# 12. Class-Wise Results

## MobileNetV2

  Emotion      Precision   Recall   F1-score   Support
  ---------- ----------- -------- ---------- ---------
  Surprise        0.5851   0.6687     0.6241       329
  Fear            0.3175   0.5405     0.4000        74
  Disgust         0.2056   0.4625     0.2846       160
  Happy           0.9248   0.5190     0.6649      1185
  Sad             0.5239   0.5732     0.5475       478
  Angry           0.3707   0.6728     0.4781       162
  Neutral         0.5566   0.5926     0.5741       680

------------------------------------------------------------------------

## DenseNet121

  Emotion      Precision   Recall   F1-score   Support
  ---------- ----------- -------- ---------- ---------
  Surprise        0.6173   0.7356     0.6713       329
  Fear            0.2727   0.5270     0.3594        74
  Disgust         0.2273   0.4688     0.3061       160
  Happy           0.9503   0.6295     0.7574      1185
  Sad             0.5871   0.6276     0.6067       478
  Angry           0.4317   0.6049     0.5039       162
  Neutral         0.6044   0.6044     0.6044       680

------------------------------------------------------------------------

## ResNet50

  Emotion      Precision   Recall   F1-score   Support
  ---------- ----------- -------- ---------- ---------
  Surprise        0.5995   0.7964     0.6841       329
  Fear            0.3150   0.5405     0.3980        74
  Disgust         0.3060   0.5125     0.3832       160
  Happy           0.9430   0.6844     0.7932      1185
  Sad             0.6795   0.6297     0.6536       478
  Angry           0.4777   0.6605     0.5544       162
  Neutral         0.6403   0.6676     0.6537       680

The results show that ResNet50 provides the strongest overall balance
among the three models.

------------------------------------------------------------------------

# 13. ROC-AUC Results

The class-wise ROC-AUC values are:

  Emotion               MobileNetV2   DenseNet121     ResNet50
  ------------------- ------------- ------------- ------------
  Surprise                   0.9133        0.9453   **0.9479**
  Fear                       0.8950        0.8882       0.8936
  Disgust                    0.8207        0.8506   **0.8792**
  Happy                      0.8991        0.9374   **0.9496**
  Sad                        0.8727        0.9025   **0.9303**
  Angry                      0.9068        0.9217   **0.9278**
  Neutral                    0.8584        0.8915   **0.9007**
  **Macro ROC-AUC**      **0.8809**    **0.9053**   **0.9184**

ResNet50 achieves the highest reported ROC-AUC for six of the seven
emotion classes. Fear is the only class where DenseNet121 has a slightly
higher AUC than ResNet50.

------------------------------------------------------------------------

# 14. Important Observation About Class Imbalance

The RAF-DB test set is strongly imbalanced.

For example:

-   Happy: 1,185 samples
-   Neutral: 680 samples
-   Sad: 478 samples
-   Surprise: 329 samples
-   Disgust: 160 samples
-   Angry: 162 samples
-   Fear: 74 samples

This matters when interpreting accuracy.

A model can achieve reasonable overall accuracy while still performing
poorly on minority classes.

Therefore, this project does not use accuracy alone to select the final
model.

Macro F1, per-class recall, confusion matrices and ROC-AUC are also
considered.

The current results show that **Fear and Disgust remain the most
difficult classes**, despite the improvements from MobileNetV2 to
ResNet50.

------------------------------------------------------------------------

# 15. Model Comparison

The current comparison can be summarized as follows:

  Criterion                        MobileNetV2   DenseNet121   ResNet50
  -------------------------------- ------------- ------------- -------------
  Accuracy                         Lowest        Medium        **Highest**
  Macro F1                         Lowest        Medium        **Highest**
  Weighted F1                      Lowest        Medium        **Highest**
  Macro ROC-AUC                    Lowest        Medium        **Highest**
  Model efficiency                 **Best**      Moderate      Lower
  Overall classification quality   Lowest        Better        **Best**

The selection therefore depends on the application.

For this project, classification quality is the primary requirement.
Consequently:

> **ResNet50 is selected as the current final emotion-classification
> model.**

MobileNetV2 remains useful as a lightweight baseline, while DenseNet121
provides an intermediate architecture for comparison.

------------------------------------------------------------------------

# 16. Inference Speed Comparison

The measured approximate classification inference results from the
current evaluation were:

  Model           Approx. Inference Time   Approx. Classification FPS
  ------------- ------------------------ ----------------------------
  MobileNetV2              5.56 ms/image                   179.78 FPS
  ResNet50                 8.34 ms/image                   119.95 FPS
  DenseNet121             14.83 ms/image                    67.43 FPS

MobileNetV2 is the fastest model.

However, the additional computational cost of ResNet50 is relatively
modest compared with DenseNet121 and provides a substantial improvement
in emotion-classification quality.

For the current project, the trade-off favors **ResNet50**.

------------------------------------------------------------------------

# 17. Training and Evaluation Visualizations

The CNN notebook generates the following visualizations for each model.

## Training and Validation Curves

Accuracy and loss are plotted together as subplots:

``` text
+----------------------+----------------------+
| Training/Validation  | Training/Validation  |
| Accuracy             | Loss                 |
+----------------------+----------------------+
```

These plots are used to identify:

-   learning progress,
-   convergence,
-   underfitting,
-   overfitting,
-   divergence between training and validation performance.

## Confusion Matrix

A seven-class confusion matrix is generated for each model.

This helps identify which emotions are commonly confused with each
other.

## ROC-AUC

One-vs-rest ROC curves are generated for all seven emotions.

The area under each curve provides a class-specific measure of
discrimination.

## Model Comparison

The final notebook compares:

-   Accuracy
-   Macro F1
-   Weighted F1
-   Macro ROC-AUC

across MobileNetV2, DenseNet121 and ResNet50.

------------------------------------------------------------------------

# 18. Real-Time Video Pipeline

The project supports prerecorded video inference.

``` text
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
ResNet50 Emotion Classification
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

The existing video-processing pipeline remains conceptually unchanged.

The CNN model used by the integration is updated from the previous
MobileNetV2 classifier to the selected **ResNet50 classifier**.

------------------------------------------------------------------------

# 19. Real-Time Webcam Detection

A standalone Python script is used for live webcam emotion detection.

The current webcam integration is running successfully.

The webcam pipeline is:

``` text
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
ResNet50 Emotion Classification
  |
  v
Emotion + Confidence
  |
  v
Live Display
```

The webcam application continuously processes incoming frames and
displays:

-   detected face bounding boxes,
-   predicted emotion,
-   emotion confidence,
-   real-time FPS.

The current `.py` integration was originally connected to the previous
CNN classifier. The remaining integration step is to replace the
MobileNetV2 model loading and preprocessing with the final ResNet50
model and its corresponding preprocessing pipeline.

The YOLO face detector remains unchanged.

------------------------------------------------------------------------

# 20. Integration Model Requirements

The final integration must preserve the same seven-class order:

``` python
CLASS_NAMES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]
```

The integration should therefore follow:

``` text
YOLO
  ↓
Face Bounding Box
  ↓
Crop Face
  ↓
Resize to ResNet50 Input Size
  ↓
ResNet50 Preprocessing
  ↓
Emotion Prediction
  ↓
CLASS_NAMES[predicted_index]
```

The model used by the final webcam application should be the saved
best-performing ResNet50 model from the CNN notebook.

------------------------------------------------------------------------

# 21. Suggested Project Structure

``` text
real-time-video-emotion-detection/
│
├── notebooks/
│   ├── cnn_emotion_classification.ipynb
│   └── yolo_face_detection.ipynb
│
├── models/
│   ├── resnet50_emotion_model.keras
│   └── yolo_face_detector_best.pt
│
├── src/
│   └── webcam_detection.py
│
├── results/
│   ├── model_comparison.png
│   ├── confusion_matrix_resnet50.png
│   ├── roc_auc_resnet50.png
│   └── training_curves/
│
├── requirements.txt
└── README.md
```

The exact filenames can be changed according to the final repository
structure.

------------------------------------------------------------------------

# 22. Technologies Used

### Programming Language

-   Python

### Deep Learning

-   TensorFlow
-   Keras
-   Ultralytics YOLO

### Computer Vision

-   OpenCV

### Data Processing

-   NumPy
-   Pandas

### Machine Learning

-   Scikit-learn

### Visualization

-   Matplotlib

### Development / Experimentation

-   Kaggle Notebook
-   Jupyter Notebook
-   VS Code

------------------------------------------------------------------------

# 23. Installation

Create a Python environment and install the required dependencies.

Example:

``` bash
pip install -r requirements.txt
```

For the local webcam application, make sure the machine has:

-   a working webcam,
-   compatible OpenCV installation,
-   TensorFlow installation,
-   Ultralytics installation,
-   the trained ResNet50 `.keras` model,
-   the trained YOLO `.pt` model.

------------------------------------------------------------------------

# 24. Running the CNN Notebook

The CNN notebook is responsible for training and evaluating the emotion
classifiers.

The general sequence is:

``` text
1. Load RAF-DB
2. Define class mapping
3. Build training and test datasets
4. Apply preprocessing and augmentation
5. Handle class imbalance
6. Build MobileNetV2
7. Train MobileNetV2
8. Evaluate MobileNetV2
9. Build DenseNet121
10. Train DenseNet121
11. Evaluate DenseNet121
12. Build ResNet50
13. Train ResNet50
14. Evaluate ResNet50
15. Generate confusion matrices
16. Generate ROC-AUC curves
17. Compare models
18. Save the best ResNet50 model
```

------------------------------------------------------------------------

# 25. Running the Webcam Application

After exporting the final ResNet50 model and keeping the existing YOLO
model, update the model paths in the webcam Python script.

Example:

``` python
CNN_MODEL_PATH = "path/to/resnet50_emotion_model.keras"
YOLO_MODEL_PATH = "path/to/yolo_face_detector_best.pt"
```

Then run:

``` bash
python webcam_detection.py
```

The webcam window displays the live detections.

Press:

``` text
Q
```

to stop the application.

------------------------------------------------------------------------

# 26. Current Project Status

  Component                             Status
  ------------------------------------- ----------------------------
  RAF-DB dataset preparation            Completed
  CNN training pipeline                 Completed
  MobileNetV2 training/evaluation       Completed
  DenseNet121 training/evaluation       Completed
  ResNet50 training/evaluation          Completed
  CNN model comparison                  Completed
  Confusion matrix evaluation           Completed
  ROC-AUC evaluation                    Completed
  YOLO face detector                    Completed
  Image inference                       Completed
  Prerecorded video inference           Completed
  Webcam application                    Working
  Webcam + final ResNet50 integration   Remaining integration step

------------------------------------------------------------------------

# 27. Current Best Model

Based on the reported results:

## ResNet50

**Test Accuracy:** 67.05%

**Macro F1:** 0.5886

**Weighted F1:** 0.6853

**Macro ROC-AUC:** 0.9184

**Test Loss:** 0.9183

ResNet50 is therefore the current selected emotion classifier for the
final system.

The YOLO face detector remains unchanged.

------------------------------------------------------------------------

# 28. Limitations

Although ResNet50 provides a significant improvement over the previous
MobileNetV2 implementation, the current results show that the problem is
not completely solved.

The most difficult classes remain:

-   Fear
-   Disgust

This is reflected in their relatively low F1-scores.

The overall accuracy of 67.05% should therefore not be interpreted as
perfect emotion recognition.

Facial emotion recognition is also affected by:

-   lighting,
-   pose,
-   occlusion,
-   face size,
-   image quality,
-   camera quality,
-   ambiguous facial expressions,
-   similarity between emotions.

The real-time system should therefore be treated as an
emotion-classification aid rather than a definitive measurement of a
person's internal emotional state.

------------------------------------------------------------------------

# 29. Future Improvements

Potential future improvements include:

-   further fine-tuning of ResNet50,
-   stronger but controlled augmentation,
-   improved handling of minority emotions,
-   targeted analysis of Fear and Disgust,
-   face alignment before classification,
-   temporal smoothing across consecutive video frames,
-   prediction stabilization to reduce frame-to-frame label changes,
-   confidence thresholding,
-   optimized inference for webcam deployment,
-   GPU acceleration,
-   model quantization or other deployment optimizations if required.

These improvements can be evaluated after establishing the current
ResNet50 + YOLO pipeline as the final baseline.

------------------------------------------------------------------------

# 30. Final System Architecture

``` text
                    ┌─────────────────────┐
                    │ Image / Video /     │
                    │ Webcam Input        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ YOLO Face Detector  │
                    │     WIDER FACE      │
                    └──────────┬──────────┘
                               │
                       Face Bounding Boxes
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Face Cropping    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ResNet50 CNN      │
                    │   RAF-DB Emotions   │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Emotion + Confidence    │
                  └────────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Annotated Live /    │
                    │ Video Output        │
                    └─────────────────────┘
```

------------------------------------------------------------------------

# 31. Conclusion

This project combines **YOLO-based face detection** with
**deep-learning-based facial emotion classification** to build an
end-to-end video emotion detection system.

The face-detection component remains based on the previously trained
YOLO model using WIDER FACE.

The emotion-classification component was rebuilt and systematically
evaluated using three CNN architectures:

-   MobileNetV2
-   DenseNet121
-   ResNet50

The current evaluation shows that **ResNet50 provides the strongest
overall classification performance**, achieving:

> **67.05% test accuracy and 0.9184 macro ROC-AUC**

while maintaining an approximate classification throughput of **119.95
FPS** in the reported benchmark.

The final system therefore uses:

``` text
YOLO Face Detector
        +
ResNet50 Emotion Classifier
        =
Real-Time Video Emotion Detection
```

The remaining integration work is to connect the final ResNet50 model to
the already-working real-time webcam application while keeping the
existing YOLO face-detection component unchanged.
