# DeepFake Audio Detection using TSSD with Attention Mechanism

As Deepfake technology advances, audio-based forgeries pose significant threats to digital security, including voice impersonation, misinformation, and fraud. This project aims to tackle these challenges by developing a robust deep learning model capable of distinguishing between real and synthetic audio.

##  Project Overview

This system leverages a **Time-Series Signal Detection (TSSD)** model integrated with an **Attention Mechanism** to enhance focus on key audio features. The model is trained on benchmark datasets and achieves high accuracy in classifying real and fake audio.

## 🎯 Objective

To build an advanced audio deepfake detection system that accurately identifies manipulated audio using time-series deep learning techniques and attention-based mechanisms.

##  Features

- **TSSD Architecture**: Designed for effective temporal feature extraction from audio signals  
- **Attention Mechanism**: Improves model sensitivity to critical audio frames  
- **MFCC Feature Extraction**: Captures spectral patterns relevant for classification  
- **High Accuracy**: Achieved **98.20%** accuracy in internal evaluations  
- **Real-time Prediction Support**: Frontend interface for audio upload and detection using Flask

## 🛠️ Tech Stack

- **Python**, **NumPy**, **Librosa** – for preprocessing and feature extraction  
- **PyTorch** – for model building and training  
- **Flask** – for creating the web interface  
- **HTML** – frontend integration  
  

## 📂 Dataset



## 🧠 Model Architecture

- **Input**: Audio signals processed to extract MFCC features  
- **TSSD Layers**: Convolutional blocks for temporal sequence modeling  
- **Residual Blocks**: For deeper representation and gradient flow  
- **Attention Layer**: Highlights important time steps  
- **Output**: Softmax layer classifying input as *Real* or *Fake*

