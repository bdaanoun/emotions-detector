import pandas as pd
from tensorflow.keras.models import load_model
from preprocessing import data_preprocess


TEST_CSV = "data/test_with_emotions.csv"

df = pd.read_csv(TEST_CSV)

model = load_model("result/model/final_emotion_model.keras")

X, y = data_preprocess(df)

loss, acc = model.evaluate(X, y)

print("loss", loss)
print("acc:", acc)