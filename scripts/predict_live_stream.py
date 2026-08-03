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

from datetime import timedelta

def get_timestamp(frame_count, fps):
    total_seconds = frame_count // fps
    return str(timedelta(seconds=total_seconds))

def predict_live_stream(model, video_path):
    cap = cv.VideoCapture(str(video_path))

    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_count = 0
    image_count = 0

    print("Reading video stream ...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process one frame every second
        if frame_count % fps == 0:
            print(f"Preprocessing ...")

            face = process_frame(frame)
            if face is not None:
                # save the image
                filename = f"preprocessing_test/image_{image_count:1d}.png"
                cv.imwrite(filename, face)
                
                # normalize and reshape the face for prediction
                processed_face = face.astype(np.float32) / 255.0
                processed_face = processed_face.reshape(1, 48, 48, 1)

                predictions = model.predict(processed_face, verbose=0)

                index = np.argmax(predictions)
                emotion = emotions[index]
                confidence = predictions[0][index] * 100

                print(f"{get_timestamp(frame_count, fps)} : {emotion}, {confidence:.2f}%")
                image_count += 1
        frame_count += 1

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    predict_live_stream(model, VIDEO_PATH)