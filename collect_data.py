import os
import cv2
import mediapipe as mp
import pandas as pd

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# -----------------------------
# Base Path
# -----------------------------
BASE_PATH = r"C:\Users\Dell\Downloads\archive\Human Action Recognition"

# Images Folder
IMAGE_FOLDER = os.path.join(BASE_PATH, "train")

# CSV File
CSV_FILE = os.path.join(BASE_PATH, "Training_set.csv")

# Output Folder
OUTPUT_FOLDER = os.path.join(BASE_PATH, "output")

# Output CSV
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "pose_data.csv")

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Image Folder :", IMAGE_FOLDER)
print("CSV File :", CSV_FILE)
print("Output File :", OUTPUT_FILE)

# Read CSV
df = pd.read_csv(CSV_FILE)

print(df.head())

print("Total Images :", len(df))

# ==========================================
# Extract Pose Landmarks from Images
# ==========================================

data = []

for index, row in df.iterrows():

    image_name = row["filename"]
    label = row["label"]

    image_path = os.path.join(IMAGE_FOLDER, image_name)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Image Not Found : {image_name}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        landmarks = []

        for landmark in results.pose_landmarks.landmark:

            landmarks.append(landmark.x)
            landmarks.append(landmark.y)
            landmarks.append(landmark.z)
            landmarks.append(landmark.visibility)

        landmarks.append(label)

        data.append(landmarks)

    else:
        print(f"No Pose Detected : {image_name}")

    # Progress after every 100 images
    if (index + 1) % 100 == 0:
        print(f"Processed {index + 1} Images")
        
        
        
# ==========================================
# Save Data into CSV
# ==========================================

columns = []

for i in range(33):
    columns.extend([
        f"x{i+1}",
        f"y{i+1}",
        f"z{i+1}",
        f"v{i+1}"
    ])

columns.append("label")

pose_df = pd.DataFrame(data, columns=columns)

pose_df.to_csv(OUTPUT_FILE, index=False)

print("\n===================================")
print("CSV File Created Successfully")
print("Saved At :", OUTPUT_FILE)
print("Total Samples :", len(pose_df))
print("===================================")