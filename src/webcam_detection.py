# ============================================================
# REAL-TIME FACIAL EMOTION DETECTION
# YOLO FACE DETECTION + AFFECTNET EMOTION CNN
#
# The emotion model contains:
# - Embedded model-specific preprocessing
# - ImageNet pretrained CNN backbone
# - Emotion classification head
#
# Webcam:
# - DirectShow backend
# - Threaded camera capture
# - Separate inference thread
# ============================================================


# ============================================================
# IMPORT LIBRARIES
# ============================================================

import cv2
import time
import threading
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess


# ============================================================
# CONFIGURATION
# ============================================================

CNN_MODEL_PATH = r"C:\Users\Harijith\Desktop\GenAI\CV_Emotion_Detection\Models\best_emotion_model.keras"
YOLO_MODEL_PATH = r"C:\Users\Harijith\Desktop\GenAI\CV_Emotion_Detection\Models\yolo_face_detector_best.pt"

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

YOLO_CONFIDENCE = 0.5

IMG_SIZE = (224, 224)

# Maximum number of faces to classify at one time
MAX_FACES = 3

# Time between inference cycles
# Increase if inference becomes too slow
INFERENCE_INTERVAL = 0.05


# ============================================================
# EMOTION CLASSES
# ============================================================

# IMPORTANT:
# This order must exactly match the class order
# used while training the AffectNet model.

EMOTION_CLASSES = [
    "Anger",
    "Contempt",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]


# ============================================================
# CUSTOM PREPROCESSING LAYERS
# ============================================================

# These layers are required because the preprocessing
# functions were embedded inside the CNN model during training.

@tf.keras.utils.register_keras_serializable(package="EmotionCNN")
class MobileNetV2Preprocess(layers.Layer):

    def call(self, inputs):
        return mobilenet_preprocess(inputs)


@tf.keras.utils.register_keras_serializable(package="EmotionCNN")
class DenseNet121Preprocess(layers.Layer):

    def call(self, inputs):
        return densenet_preprocess(inputs)


@tf.keras.utils.register_keras_serializable(package="EmotionCNN")
class ResNet50Preprocess(layers.Layer):

    def call(self, inputs):
        return resnet_preprocess(inputs)


# ============================================================
# APPLICATION INFORMATION
# ============================================================

print("=" * 65)
print("REAL-TIME FACIAL EMOTION DETECTION")
print("YOLO FACE DETECTION + AFFECTNET EMOTION CNN")
print("=" * 65)

print(f"\nTensorFlow Version: {tf.__version__}")

print("\nAvailable TensorFlow GPU Devices:")
print(tf.config.list_physical_devices("GPU"))

print("\nEmotion Classes:")

for index, emotion in enumerate(EMOTION_CLASSES):
    print(f"{index}: {emotion}")


# ============================================================
# LOAD EMOTION CLASSIFICATION MODEL
# ============================================================

print("\nLoading emotion classification model...")

emotion_model = keras.models.load_model(
    CNN_MODEL_PATH,
    custom_objects={
        "MobileNetV2Preprocess": MobileNetV2Preprocess,
        "DenseNet121Preprocess": DenseNet121Preprocess,
        "ResNet50Preprocess": ResNet50Preprocess
    }
)

print("Emotion classification model loaded successfully.")

print("\nModel Input Shape:")
print(emotion_model.input_shape)

print("\nModel Output Shape:")
print(emotion_model.output_shape)


# ============================================================
# LOAD YOLO FACE DETECTOR
# ============================================================

print("\nLoading YOLO face detection model...")

face_detector = YOLO(YOLO_MODEL_PATH)

print("YOLO face detector loaded successfully.")


# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(face_image):

    # Convert OpenCV BGR image to RGB
    face_rgb = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    # Resize face to CNN input size
    face_resized = cv2.resize(
        face_rgb,
        IMG_SIZE
    )

    # Convert to float32
    face_array = np.asarray(
        face_resized,
        dtype=np.float32
    )

    # Add batch dimension
    face_array = np.expand_dims(
        face_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT apply MobileNetV2, DenseNet121 or ResNet50
    # preprocessing here.
    #
    # Preprocessing is already embedded inside the saved model.

    predictions = emotion_model(
        face_array,
        training=False
    ).numpy()[0]

    # Get highest probability index
    predicted_index = int(
        np.argmax(predictions)
    )

    # Get confidence score
    confidence = float(
        predictions[predicted_index]
    )

    # Get emotion name
    emotion = EMOTION_CLASSES[
        predicted_index
    ]

    return emotion, confidence, predictions


# ============================================================
# YOLO FACE DETECTION + EMOTION CLASSIFICATION
# ============================================================

def detect_faces_and_emotions(image):

    # Create output copy
    output = image.copy()

    # Run YOLO face detection
    results = face_detector(
        image,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    # Track number of processed faces
    face_count = 0

    # Process YOLO results
    for result in results:

        # Skip if no bounding boxes
        if result.boxes is None:
            continue

        # Process each detected face
        for box in result.boxes:

            # Limit number of faces
            if face_count >= MAX_FACES:
                break

            # Detection confidence
            detection_confidence = float(
                box.conf[0]
            )

            # Skip low-confidence detections
            if detection_confidence < YOLO_CONFIDENCE:
                continue

            # Get bounding box coordinates
            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            # Keep bounding box inside image
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )

            # Validate bounding box
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop detected face
            face = image[
                y1:y2,
                x1:x2
            ]

            # Skip empty crops
            if face.size == 0:
                continue

            try:

                # Predict emotion
                emotion, emotion_confidence, predictions = (
                    predict_emotion(face)
                )

            except Exception as error:

                print(
                    f"Emotion prediction error: {error}"
                )

                continue

            # Create display label
            label = (
                f"{emotion} "
                f"{emotion_confidence * 100:.1f}%"
            )

            # Draw face bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Calculate label background size
            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            text_width = text_size[0]

            # Label background coordinates
            label_y1 = max(
                0,
                y1 - 32
            )

            label_x2 = min(
                output.shape[1],
                x1 + text_width + 15
            )

            # Draw label background
            cv2.rectangle(
                output,
                (x1, label_y1),
                (label_x2, y1),
                (0, 255, 0),
                -1
            )

            # Draw emotion label
            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Increase processed face count
            face_count += 1

    return output


# ============================================================
# GLOBAL THREAD VARIABLES
# ============================================================

latest_frame = None

latest_result = None

frame_lock = threading.Lock()

result_lock = threading.Lock()

camera_running = True

inference_running = True


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_thread():

    global latest_frame
    global camera_running

    print("\nOpening webcam using DirectShow...")

    # DirectShow is used because it worked correctly
    # during the webcam diagnostic test.

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    # Check camera
    if not camera.isOpened():

        print(
            "ERROR: DirectShow could not open webcam."
        )

        camera_running = False

        return

    # Set resolution
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # Try setting MJPG format
    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    # Request camera FPS
    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    # Reduce internal camera buffering
    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("DirectShow webcam opened successfully.")
    print("Camera thread started.")

    # Track repeated frame failures
    failed_frames = 0

    while camera_running:

        # Read frame
        ret, frame = camera.read()

        # Handle failed frame
        if not ret or frame is None:

            failed_frames += 1

            # Avoid flooding terminal
            if failed_frames % 50 == 0:

                print(
                    "Warning: Failed to receive webcam frame."
                )

            time.sleep(0.01)

            continue

        # Reset failure counter
        failed_frames = 0

        # Mirror webcam
        frame = cv2.flip(
            frame,
            1
        )

        # Store newest frame
        with frame_lock:

            latest_frame = frame.copy()

    # Release camera
    camera.release()

    print("Camera thread stopped.")


# ============================================================
# INFERENCE THREAD
# ============================================================

def inference_thread():

    global latest_result
    global inference_running

    print("Inference thread started.")

    last_processed_time = 0

    while inference_running:

        # Get newest webcam frame
        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        # Wait if camera frame is unavailable
        if frame is None:

            time.sleep(0.01)

            continue

        # Current time
        current_time = time.time()

        # Control inference frequency
        if (
            current_time
            -
            last_processed_time
            <
            INFERENCE_INTERVAL
        ):

            time.sleep(0.005)

            continue

        # Update processing time
        last_processed_time = current_time

        try:

            # Run YOLO + emotion CNN
            result = detect_faces_and_emotions(
                frame
            )

            # Store latest processed frame
            with result_lock:

                latest_result = result

        except Exception as error:

            print(
                f"Inference error: {error}"
            )

            time.sleep(0.01)

    print("Inference thread stopped.")


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_worker = threading.Thread(
    target=camera_thread,
    daemon=True
)

camera_worker.start()


# ============================================================
# WAIT FOR WEBCAM FRAMES
# ============================================================

print("\nWaiting for webcam frames...")

timeout = time.time() + 10

while True:

    # Check if frame is available
    with frame_lock:

        frame_available = (
            latest_frame is not None
        )

    # Exit if frame received
    if frame_available:

        break

    # Stop if timeout occurs
    if time.time() > timeout:

        camera_running = False

        raise RuntimeError(
            "Webcam opened but no frames were received."
        )

    time.sleep(0.05)


print("Webcam frames are being received successfully.")


# ============================================================
# START INFERENCE THREAD
# ============================================================

inference_worker = threading.Thread(
    target=inference_thread,
    daemon=True
)

inference_worker.start()


# ============================================================
# APPLICATION START MESSAGE
# ============================================================

print()
print("=" * 65)
print("LIVE EMOTION DETECTION STARTED")
print("=" * 65)
print()
print("Pipeline:")
print("Webcam -> YOLO Face Detection -> Emotion CNN")
print()
print("Press Q or ESC to exit.")
print()


# ============================================================
# DISPLAY VARIABLES
# ============================================================

display_previous_time = time.time()

display_fps = 0.0


# ============================================================
# MAIN DISPLAY LOOP
# ============================================================

try:

    while True:

        # ----------------------------------------------------
        # GET LATEST CAMERA FRAME
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                frame = None

            else:

                frame = latest_frame.copy()

        # Skip if unavailable
        if frame is None:

            time.sleep(0.01)

            continue


        # ----------------------------------------------------
        # GET LATEST INFERENCE RESULT
        # ----------------------------------------------------

        with result_lock:

            if latest_result is not None:

                display_frame = latest_result.copy()

            else:

                display_frame = frame.copy()


        # ----------------------------------------------------
        # CALCULATE DISPLAY FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time
            -
            display_previous_time
        )

        display_previous_time = current_time

        if elapsed > 0:

            instant_fps = (
                1.0 / elapsed
            )

            # Smooth FPS value
            display_fps = (
                0.9 * display_fps
                +
                0.1 * instant_fps
            )


        # ----------------------------------------------------
        # DISPLAY FPS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"Display FPS: {display_fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # DISPLAY SYSTEM STATUS
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "YOLO + Emotion CNN",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display_frame,
            "Press Q or ESC to quit",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # DISPLAY VIDEO
        # ----------------------------------------------------

        cv2.imshow(
            "Real-Time Facial Emotion Detection",
            display_frame
        )


        # ----------------------------------------------------
        # KEYBOARD INPUT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Quit using Q or ESC
        if key == ord("q") or key == 27:

            print(
                "\nStopping application..."
            )

            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    # Stop camera thread
    camera_running = False

    # Stop inference thread
    inference_running = False

    print("Stopping threads...")

    # Wait for camera thread
    camera_worker.join(
        timeout=3
    )

    # Wait for inference thread
    inference_worker.join(
        timeout=3
    )

    # Close OpenCV windows
    cv2.destroyAllWindows()

    print("Application stopped successfully.")