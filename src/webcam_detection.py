# ============================================================
# REAL-TIME VIDEO EMOTION DETECTION
# YOLO FACE DETECTION + RESNET50 EMOTION CLASSIFICATION
#
# Threaded Webcam + YOLO + ResNet50
# ============================================================

import cv2
import time
import threading
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CNN_MODEL_PATH = r"C:\Users\Harijith\Downloads\best_emotion_model.keras"
YOLO_MODEL_PATH = r"C:\Users\Harijith\Downloads\yolo_face_detector_best.pt"

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

YOLO_CONFIDENCE = 0.5

IMG_SIZE = (224, 224)

# Maximum number of faces to classify
MAX_FACES = 3

# Time between emotion inference operations.
#
# Lower value = more frequent inference but more GPU/CPU usage.
# Higher value = less computation and smoother webcam display.
INFERENCE_INTERVAL = 0.10


# ============================================================
# EMOTION CLASSES
# ============================================================

EMOTION_CLASSES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happy",
    "Sad",
    "Angry",
    "Neutral"
]


# ============================================================
# APPLICATION INFORMATION
# ============================================================

print("=" * 70)
print("REAL-TIME VIDEO EMOTION DETECTION")
print("YOLO FACE DETECTION + RESNET50 EMOTION CLASSIFICATION")
print("=" * 70)

print("\nTensorFlow version:")
print(tf.__version__)

print("\nTensorFlow GPU devices:")
print(tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD RESNET50 EMOTION MODEL
# ============================================================

print("\nLoading ResNet50 emotion model...")

# The model was created using:
#
# layers.Lambda(
#     resnet_preprocess
# )
#
# Therefore resnet_preprocess must be available while
# loading the saved Keras model.

emotion_model = load_model(
    CNN_MODEL_PATH,
    custom_objects={
        "preprocess_input": resnet_preprocess,
        "resnet_preprocess": resnet_preprocess
    }
)

print("ResNet50 emotion model loaded successfully.")


# ============================================================
# LOAD YOLO FACE DETECTOR
# ============================================================

print("\nLoading YOLO face detector...")

face_detector = YOLO(YOLO_MODEL_PATH)

print("YOLO face detector loaded successfully.")


# ============================================================
# EMOTION PREDICTION FUNCTION
# ============================================================

def predict_emotion(face_image):
    """
    Predict emotion from a cropped face.

    Input:
        face_image:
            OpenCV BGR face image.

    Returns:
        emotion:
            Predicted emotion class.

        confidence:
            Prediction probability.
    """

    # --------------------------------------------------------
    # Convert BGR to RGB
    # --------------------------------------------------------

    face_rgb = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Resize to ResNet50 input size
    # --------------------------------------------------------

    face_resized = cv2.resize(
        face_rgb,
        IMG_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    face_array = np.asarray(
        face_resized,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    face_array = np.expand_dims(
        face_array,
        axis=0
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Do NOT manually apply MobileNetV2 preprocessing.
    #
    # The ResNet50 preprocessing is already part of the
    # saved model through the Lambda layer.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    predictions = emotion_model(
        face_array,
        training=False
    ).numpy()[0]

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    emotion = EMOTION_CLASSES[
        predicted_index
    ]

    return emotion, confidence


# ============================================================
# YOLO + RESNET50 EMOTION DETECTION
# ============================================================

def detect_faces_and_emotions(image):
    """
    Detect faces using YOLO and classify each face using
    the ResNet50 emotion classifier.
    """

    output = image.copy()

    # --------------------------------------------------------
    # Run YOLO face detection
    # --------------------------------------------------------

    results = face_detector(
        image,
        conf=YOLO_CONFIDENCE,
        verbose=False
    )

    face_count = 0

    # --------------------------------------------------------
    # Process detected faces
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # ------------------------------------------------
            # Maximum face limit
            # ------------------------------------------------

            if face_count >= MAX_FACES:
                break

            # ------------------------------------------------
            # YOLO detection confidence
            # ------------------------------------------------

            detection_confidence = float(
                box.conf[0]
            )

            if detection_confidence < YOLO_CONFIDENCE:
                continue

            # ------------------------------------------------
            # Bounding box coordinates
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = map(
                int,
                [x1, y1, x2, y2]
            )

            # ------------------------------------------------
            # Keep coordinates inside frame
            # ------------------------------------------------

            x1 = max(
                0,
                x1
            )

            y1 = max(
                0,
                y1
            )

            x2 = min(
                image.shape[1],
                x2
            )

            y2 = min(
                image.shape[0],
                y2
            )

            # ------------------------------------------------
            # Validate bounding box
            # ------------------------------------------------

            if x2 <= x1 or y2 <= y1:
                continue

            # ------------------------------------------------
            # Crop face
            # ------------------------------------------------

            face = image[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # Emotion classification
            # ------------------------------------------------

            try:

                emotion, emotion_confidence = (
                    predict_emotion(face)
                )

            except Exception as e:

                print(
                    "Emotion prediction error:",
                    e
                )

                continue

            # ------------------------------------------------
            # Create label
            # ------------------------------------------------

            label = (
                f"{emotion} "
                f"{emotion_confidence * 100:.1f}%"
            )

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            label_y1 = max(
                0,
                y1 - 30
            )

            label_y2 = y1

            label_width = 190

            cv2.rectangle(
                output,
                (x1, label_y1),
                (
                    x1 + label_width,
                    label_y2
                ),
                (0, 255, 0),
                -1
            )

            # ------------------------------------------------
            # Draw emotion label
            # ------------------------------------------------

            cv2.putText(
                output,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

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

    # --------------------------------------------------------
    # IMPORTANT
    #
    # DirectShow was confirmed to work with your webcam.
    # Therefore use CAP_DSHOW directly instead of trying
    # MSMF first.
    # --------------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam using DirectShow."
        )

        camera_running = False

        return

    # --------------------------------------------------------
    # Camera resolution
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # --------------------------------------------------------
    # Use MJPEG
    #
    # This usually works well with USB webcams on Windows.
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(
            *"MJPG"
        )
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        CAMERA_FPS
    )

    # --------------------------------------------------------
    # Allow camera initialization
    # --------------------------------------------------------

    time.sleep(0.5)

    # --------------------------------------------------------
    # Verify that actual frames are received
    # --------------------------------------------------------

    print("Testing webcam frames...")

    frame_received = False

    for _ in range(30):

        ret, frame = camera.read()

        if (
            ret
            and frame is not None
            and frame.size > 0
        ):

            frame_received = True

            break

        time.sleep(0.05)

    if not frame_received:

        print(
            "ERROR: Webcam opened but no frames were received."
        )

        camera.release()

        camera_running = False

        return

    print(
        "Webcam opened successfully."
    )

    print(
        "Resolution:",
        frame.shape[1],
        "x",
        frame.shape[0]
    )

    # --------------------------------------------------------
    # Main camera loop
    # --------------------------------------------------------

    while camera_running:

        ret, frame = camera.read()

        if not ret or frame is None:

            print(
                "Warning: failed to read webcam frame."
            )

            time.sleep(0.01)

            continue

        # ----------------------------------------------------
        # Mirror webcam
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # Store newest frame
        # ----------------------------------------------------

        with frame_lock:

            latest_frame = frame.copy()

    # --------------------------------------------------------
    # Release camera
    # --------------------------------------------------------

    camera.release()

    print(
        "Camera thread stopped."
    )


# ============================================================
# INFERENCE THREAD
# ============================================================

def inference_thread():

    global latest_result
    global inference_running

    print(
        "Inference thread started."
    )

    last_processed_time = 0

    while inference_running:

        # ----------------------------------------------------
        # Get newest frame
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                time.sleep(0.01)

                continue

            frame = latest_frame.copy()

        # ----------------------------------------------------
        # Control inference frequency
        # ----------------------------------------------------

        current_time = time.time()

        if (
            current_time
            - last_processed_time
            <
            INFERENCE_INTERVAL
        ):

            time.sleep(0.005)

            continue

        last_processed_time = current_time

        # ----------------------------------------------------
        # YOLO + ResNet50
        # ----------------------------------------------------

        try:

            result = detect_faces_and_emotions(
                frame
            )

            # ------------------------------------------------
            # Store newest inference result
            # ------------------------------------------------

            with result_lock:

                latest_result = result

        except Exception as e:

            print(
                "Inference error:",
                e
            )

    print(
        "Inference thread stopped."
    )


# ============================================================
# START CAMERA THREAD
# ============================================================

camera_worker = threading.Thread(
    target=camera_thread,
    daemon=True
)

camera_worker.start()


# ============================================================
# WAIT FOR CAMERA
# ============================================================

print(
    "Waiting for webcam..."
)

timeout = time.time() + 10

while latest_frame is None:

    if not camera_running:

        raise RuntimeError(
            "Webcam could not be started."
        )

    if time.time() > timeout:

        camera_running = False

        raise RuntimeError(
            "Webcam opened but no frames were received."
        )

    time.sleep(0.05)


print(
    "Receiving webcam frames."
)


# ============================================================
# START INFERENCE THREAD
# ============================================================

inference_worker = threading.Thread(
    target=inference_thread,
    daemon=True
)

inference_worker.start()


# ============================================================
# DISPLAY LOOP
# ============================================================

print()
print("=" * 70)
print("LIVE EMOTION DETECTION STARTED")
print("=" * 70)
print()
print("Pipeline:")
print("Webcam -> YOLO Face Detection -> ResNet50 Emotion Classification")
print()
print("Press Q or ESC to exit.")
print()


display_previous_time = time.time()

display_fps = 0.0

last_display_frame = None


try:

    while True:

        # ----------------------------------------------------
        # Get latest camera frame
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                continue

            frame = latest_frame.copy()

        # ----------------------------------------------------
        # Get latest inference result
        # ----------------------------------------------------

        with result_lock:

            if latest_result is not None:

                display_frame = latest_result.copy()

            else:

                display_frame = frame.copy()

        # ----------------------------------------------------
        # Calculate display FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time
            - display_previous_time
        )

        display_previous_time = current_time

        if elapsed > 0:

            instant_fps = 1.0 / elapsed

            display_fps = (
                0.9 * display_fps
                +
                0.1 * instant_fps
            )

        # ----------------------------------------------------
        # Display FPS
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
        # Display model information
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "YOLO + ResNet50",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Exit information
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "Press Q to quit",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Display frame
        # ----------------------------------------------------

        cv2.imshow(
            "Real-Time Emotion Detection",
            display_frame
        )

        # ----------------------------------------------------
        # Keyboard input
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if (
            key == ord("q")
            or key == 27
        ):

            print(
                "\nStopping application..."
            )

            break


finally:

    # ========================================================
    # STOP THREADS
    # ========================================================

    camera_running = False
    inference_running = False

    # --------------------------------------------------------
    # Wait for camera thread
    # --------------------------------------------------------
    camera_worker.join(timeout=2)

    # --------------------------------------------------------
    # Wait for inference thread
    # --------------------------------------------------------
    inference_worker.join(timeout=2)
    # --------------------------------------------------------
    # Close OpenCV windows
    # --------------------------------------------------------
    cv2.destroyAllWindows()
    print("Application stopped.")