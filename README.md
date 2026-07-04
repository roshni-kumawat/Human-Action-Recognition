# Human Action Recognition Using MediaPipe and SVM

## Overview

The **Human Action Recognition Using MediaPipe and SVM** project is a computer vision and machine learning application developed using **Python, MediaPipe, OpenCV, and Support Vector Machine (SVM)**. It extracts human body landmarks from images and recognizes different human actions. The project demonstrates pose estimation, feature extraction, model training, and real-time action prediction using a webcam.

---

## Features

- Human pose detection using MediaPipe
- Extraction of 33 body landmarks
- SVM-based action classification
- Real-time webcam prediction
- Fast and lightweight implementation
- Easy-to-use machine learning pipeline

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- Pandas
- NumPy
- Joblib
- TensorFlow

---

## Dataset

**Human Action Recognition Dataset**

### Sample Actions

- Running
- Cycling
- Dancing
- Sitting
- ...and 11 more action classes

---

## How It Works

1. Install the required Python libraries.
2. Prepare the Human Action Recognition dataset.
3. Extract pose landmarks using MediaPipe.
4. Generate the feature dataset (`pose_data.csv`).
5. Train the SVM model using `train_model.py`.
6. Save the trained model as `Human_pose.pkl`.
7. Run `predict.py` to perform real-time human action recognition using a webcam.

---

## Applications

- Human Activity Recognition
- Smart Surveillance
- Fitness Monitoring
- Healthcare
- Human-Computer Interaction
- Computer Vision Research

---

## License

This project is developed for educational and learning purposes under the MIT License.
