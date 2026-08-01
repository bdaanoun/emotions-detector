import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preprocessing import process_frame

VIDEO_PATH = REPO_ROOT / "preprocessing_test" / "input_video.mp4"
MODEL_PATH = REPO_ROOT / "results" / "model" / "final_emotion_model.keras"
model = load_model(str(MODEL_PATH))

emotions = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


def predict_live_stream(model, video_path):
    cap = cv.VideoCapture(str(video_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face = process_frame(frame)
        print("Processing frame...")
        if face is None:
            continue

        # process the face for CNN
        processed_face = face.astype(np.float32) / 255.0
        processed_face = processed_face.reshape(1, 48, 48, 1)

        # predict the emotion
        print("Predicting...")
        predictions = model.predict(processed_face, verbose=0)

        # convert predictions to emotion label
        index = np.argmax(predictions)
        emotion = emotions[index]

        confidence = predictions[0][index] *100

        print(f"{emotion}, {confidence:.2f}%")
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    predict_live_stream(model, VIDEO_PATH)