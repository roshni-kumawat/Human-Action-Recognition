import cv2
import mediapipe as mp
import joblib

# ===============================
# Load Trained Model
# ===============================

MODEL_PATH = r"C:\Users\Dell\Downloads\archive\Human Action Recognition\models\Human_pose.pkl"

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully!")

# ===============================
# Initialize MediaPipe
# ===============================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# Start Webcam
# ===============================

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks:

        landmarks = []

        for landmark in results.pose_landmarks.landmark:

            landmarks.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])

        if len(landmarks) == 132:

            prediction = model.predict([landmarks])[0]

            cv2.putText(
                frame,
                prediction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Human Action Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
