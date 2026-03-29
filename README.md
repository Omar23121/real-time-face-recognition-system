# Real-Time AI Face Recognition System

A real-time face recognition system built using DeepFace, OpenCV, and ArcFace embeddings.  
This project performs accurate identity matching from images, webcam streams, and video files.

---

# Features

- Real-time face recognition via webcam  
- Image-based face recognition  
- Video file processing with annotated output  
- Deep learning-based face embeddings (ArcFace)  
- Multi-identity recognition using custom dataset  
- Unknown face detection  
- FPS (performance) display in real-time  
- Automatic logging of detections (CSV format)  
- Modular and scalable architecture  

---

# Tech Stack

- Python  
- OpenCV  
- DeepFace  
- NumPy  
- tf-keras  

---

# Project Structure

face_recognition_app/

├── ai_face_recognition.py  
├── requirements.txt  
├── known_faces/  
│   ├── Elon Musk/  
│   ├── Tom Cruise/  
│   └── ...  
├── output/  
├── logs/  
└── test.jpg  

---

# Installation

python -m pip install -r requirements.txt

---

# Usage

## Image Mode

python ai_face_recognition.py image test.jpg

---

## Webcam Mode

python ai_face_recognition.py webcam

Press 'q' to quit.

---

## Video Mode

python ai_face_recognition.py video sample.mp4

---

# Output

Annotated images saved to:

output/recognized_output.jpg

Annotated videos saved to:

output/recognized_video.mp4

Detection logs saved to:

logs/recognition_log.csv

Example log entry:

2026-03-29 14:22:10,webcam,Elon Musk,0.2184,webcam

---

# How It Works

1. Faces are detected using OpenCV  
2. DeepFace generates facial embeddings using the ArcFace model  
3. Embeddings are compared using cosine similarity  
4. The closest match is selected based on a distance threshold  
5. If no match is close enough → labeled as Unknown  

---

#  Key Concepts

- Face Embeddings  
- Cosine Distance Similarity  
- Real-time Computer Vision  
- Deep Learning-based Recognition  

---

# Challenges & Solutions

## Problem:
Initial implementation using traditional methods (LBPH) produced inaccurate results.

## Solution:
Switched to DeepFace with ArcFace embeddings, significantly improving recognition accuracy.

---

## Problem:
Dataset contained low-quality images that caused detection failures.

## Solution:
Filtered out images where faces could not be detected to improve model reliability.

---

## Problem:
DeepFace caching caused outdated embeddings to be used.

## Solution:
Cleared .pkl cache files to force recomputation of embeddings.

---

# Performance Notes

- Real-time FPS displayed in webcam/video mode  
- Frame skipping implemented for performance optimization  
- Recognition accuracy depends on dataset quality


---

# Dataset

The dataset used for this project was sourced from Kaggle:

https://www.kaggle.com/datasets/anku5hk/5-faces-dataset/data

Only a subset of high-quality images was selected and cleaned to improve recognition accuracy.

---

# Future Improvements

- GUI interface (Streamlit or web app)  
- Face tracking to reduce flickering  
- Faster inference using embedding caching  
- Multi-face independent recognition  
- Deployment as a web service  

---

# Why This Project Matters

This project demonstrates:

- Practical application of AI and computer vision  
- Real-time system design  
- Debugging and optimization of ML pipelines  
- Handling real-world data issues  

---

# Author

Omar Almahmoud
