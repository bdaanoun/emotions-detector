# training the cnn model
from preprocessing import data_preprocess
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

import pandas as pd
import os
import datetime

df = pd.read_csv("data/train.csv")

pixels, y = data_preprocess(df)

X_train, X_test, y_train, y_test = train_test_split(pixels, y, test_size=0.2, random_state=42, stratify=y)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
])

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5, 
    restore_best_weights=True)

model = Sequential([
    Input(shape=(48,48,1)),
    data_augmentation,
    Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(7, activation="softmax")
])

os.makedirs("logs/fit", exist_ok=True)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics= ["accuracy"],
)


history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test),callbacks=[tensorboard_callback, early_stop])

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


with open("final_emotion_model_arch.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

    
os.makedirs("results/model", exist_ok=True)
model.save("results/model/final_emotion_model.keras")