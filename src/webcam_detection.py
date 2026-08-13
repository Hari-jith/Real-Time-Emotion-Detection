# ============================================================
# REAL-TIME VIDEO EMOTION DETECTION
# YOLO + MobileNetV2
#
# Threaded Webcam + Inference Pipeline
# ============================================================

import cv2
import time
import threading
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

CNN_MODEL_PATH = (r"C:\Users\Harijith\Downloads\best_emotion_model.keras")
YOLO_MODEL_PATH = (r"C:\Users\Harijith\Downloads\yolo_face_detector_best.pt")

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
YOLO_CONFIDENCE = 0.5
IMG_SIZE = (224, 224)

# Maximum number of faces to classify
MAX_FACES = 3

# Run inference approximately every N seconds.
# Increase this if inference is very slow.
INFERENCE_INTERVAL = 0.05


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
# GPU INFORMATION
# ============================================================

print("=" * 60)
print("REAL-TIME VIDEO EMOTION DETECTION")
print("=" * 60)

print("\nTensorFlow version:")
print(tf.__version__)

print("\nTensorFlow GPU devices:")
print(tf.config.list_physical_devices("GPU"))


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

print("\nLoading MobileNetV2...")
emotion_model = load_model(CNN_MODEL_PATH)
print("MobileNetV2 loaded.")

# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO...")
face_detector = YOLO(YOLO_MODEL_PATH)
print("YOLO loaded.")

# ============================================================
# EMOTION PREDICTION
# ============================================================

def predict_emotion(face_image):

    # BGR -> RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Resize
    face_resized = cv2.resize(face_rgb, IMG_SIZE)

    # Float32
    face_array = np.asarray(face_resized, dtype=np.float32)

    # Batch dimension
    face_array = np.expand_dims(face_array, axis=0)

    # MobileNetV2 preprocessing
    face_array = preprocess_input(face_array)

    # Prediction
    predictions = emotion_model(face_array, training=False).numpy()[0]
    predicted_index = np.argmax(predictions)
    confidence = float(predictions[predicted_index])
    emotion = EMOTION_CLASSES[predicted_index]
    return emotion, confidence


# ============================================================
# YOLO + EMOTION DETECTION
# ============================================================

def detect_faces_and_emotions(image):

    output = image.copy()

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    results = face_detector(image, conf=YOLO_CONFIDENCE, verbose=False)
    face_count = 0

    # --------------------------------------------------------
    # Process detections
    # --------------------------------------------------------

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            if face_count >= MAX_FACES:
                break

            # Detection confidence
            detection_confidence = float(box.conf[0])
            if detection_confidence < YOLO_CONFIDENCE:
                continue

            # Bounding box
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Keep inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # Crop face
            face = image[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            # ------------------------------------------------
            # Emotion classification
            # ------------------------------------------------

            emotion, emotion_confidence = (predict_emotion(face))
            label = (
                f"{emotion} "
                f"{emotion_confidence * 100:.1f}%"
            )

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            cv2.rectangle(
                output,
                (x1, max(0, y1 - 30)),
                (x1 + 190, y1),
                (0, 255, 0),
                -1
            )

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

    print("\nOpening webcam...")

    # Try Microsoft Media Foundation first
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_MSMF)

    # Fallback to DirectShow
    if not camera.isOpened():

        print("MSMF failed. Trying DirectShow...")
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")
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

    print("Webcam opened successfully.")
    while camera_running:
        ret, frame = camera.read()
        if not ret:

            print("Warning: failed to read webcam frame.")
            time.sleep(0.01)

            continue

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # Store newest frame
        with frame_lock:
            latest_frame = frame.copy()

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

        # ----------------------------------------------------
        # Get latest camera frame
        # ----------------------------------------------------

        with frame_lock:

            if latest_frame is None:

                time.sleep(0.01)

                continue

            frame = latest_frame.copy()

        # ----------------------------------------------------
        # Control inference rate
        # ----------------------------------------------------

        current_time = time.time()

        if (
            current_time
            -
            last_processed_time
            <
            INFERENCE_INTERVAL
        ):

            time.sleep(0.005)

            continue

        last_processed_time = current_time

        # ----------------------------------------------------
        # YOLO + MobileNetV2
        # ----------------------------------------------------

        try:

            result = detect_faces_and_emotions(
                frame
            )

            # Store newest result
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

print("Waiting for webcam...")

timeout = time.time() + 10

while latest_frame is None:

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
print("=" * 60)
print("LIVE EMOTION DETECTION STARTED")
print("=" * 60)
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
        # Calculate DISPLAY FPS
        # ----------------------------------------------------

        current_time = time.time()

        elapsed = (
            current_time
            -
            display_previous_time
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
        # FPS
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
        # Status
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            "YOLO + MobileNetV2",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


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
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "Real-Time Emotion Detection",
            display_frame
        )


        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:

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

    camera_worker.join(
        timeout=2
    )

    # --------------------------------------------------------
    # Wait for inference thread
    # --------------------------------------------------------

    inference_worker.join(
        timeout=2
    )

    # --------------------------------------------------------
    # Destroy windows
    # --------------------------------------------------------

    cv2.destroyAllWindows()
    print("Application stopped.")