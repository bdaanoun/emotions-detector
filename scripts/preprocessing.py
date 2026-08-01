import pandas as pd
import numpy as np

import cv2 as cv
from scipy.datasets import face

def data_preprocess(df):
    pixels = df["pixels"]
    y = df["emotion"]

    pixels = pixels.apply(lambda x: np.fromstring(x, sep=' '))
    pixels = np.stack(pixels.values)
    pixels = pixels.reshape(-1, 48, 48, 1)
    pixels = pixels.astype(np.float32) / 255.0

    return pixels, y


# Load the face detector
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def process_frame(frame):
        cap = cv.VideoCapture("preprocessing_test/input_video.mp4")
        fps = int(cap.get(cv.CAP_PROP_FPS))
        # Convert frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]  # Crop the face

        # Resize to 48x48
        face = cv.resize(face, (48, 48))

        return face