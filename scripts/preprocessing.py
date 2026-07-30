import pandas as pd
import numpy as np

def data_preprocess(df):
    pixels = df["pixels"]
    y = df["emotion"]

    pixels = pixels.apply(lambda x: np.fromstring(x, sep=' '))
    pixels = np.stack(pixels.values)
    pixels = pixels.reshape(-1, 48, 48, 1)
    pixels = pixels.astype(np.float32) / 255.0

    return pixels, y