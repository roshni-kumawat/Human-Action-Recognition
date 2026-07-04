import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# File Paths
# ==========================================

CSV_FILE = r"C:\Users\Dell\Downloads\archive\Human Action Recognition\output\pose_data.csv"

MODEL_FOLDER = r"C:\Users\Dell\Downloads\archive\Human Action Recognition\models"

# Agar models folder nahi hai to bana do
os.makedirs(MODEL_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_FOLDER, "Human_pose.pkl")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(CSV_FILE)

print("Dataset Shape :", df.shape)

# ==========================================
# Features and Labels
# ==========================================

X = df.drop("label", axis=1)
y = df["label"]

print("Features Shape :", X.shape)
print("Labels Shape :", y.shape)

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ==========================================
# Train SVM Model
# ==========================================

print("\nTraining SVM Model...")

model = SVC(kernel="rbf")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, MODEL_PATH)

print("\n====================================")
print("Model Saved Successfully!")
print("Model Location :", MODEL_PATH)
print("====================================")