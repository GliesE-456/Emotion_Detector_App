import pandas as pd
from emotion_model import analyze_text

# Load your labeled test data
df = pd.read_csv("test_data.csv")  # columns: text, label

correct = 0
total = len(df)

for _, row in df.iterrows():
    prediction = analyze_text(row["text"])
    pred_label = prediction.get("label", "neutral")
    if pred_label == row["label"]:
        correct += 1

accuracy = correct / total if total > 0 else 0
print(f"Model accuracy: {accuracy:.2%}")