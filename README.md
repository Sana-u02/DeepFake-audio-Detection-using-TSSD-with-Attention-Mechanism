# DeepFake Audio Detection using TSSD with Attention Mechanism

As Deepfake technology advances, audio-based forgeries pose significant threats to digital security, including voice impersonation, misinformation, and fraud. This project aims to tackle these challenges by developing a robust deep learning model capable of distinguishing between real and synthetic audio.

## Project Overview

This system leverages a **Time Domain Synthetic Speech Detection (TSSD)** model integrated with an **Attention Mechanism** to enhance focus on key audio features. The model is trained on benchmark datasets and achieves high accuracy in classifying real and fake audio.

## 🎯 Objective

To build an advanced audio deepfake detection system that accurately identifies manipulated audio using time-series deep learning techniques and attention-based mechanisms.

## Features

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

- **Real audio clips**: 10,000 samples from public speech datasets  
- **Fake audio clips**: 10,000 clips generated using TTS models like WaveNet  
- Clips are 2–3 seconds long WAV files

---

## 🧠 Model Architecture

- Input: Audio converted to MFCC, Mel Spectrogram, and other features  
- Temporal feature extraction with 1D CNN + Residual blocks (TSSD)  
- Multi-head attention highlights critical time frames  
- Fully connected layers for final classification  
- Output: Real vs Fake audio via Softmax

<img src="https://github.com/user-attachments/assets/70277c28-c765-463b-b8a5-85a4fb00052e" alt="new_arc" width="400"/>



---

## 📊 Performance

| Model                 | Accuracy | EER    | T-DCF  |
|-----------------------|----------|--------|--------|
| **TSSD + Attention**  | 98.20%   | 0.0505 | 0.0044 |
| TSSD              | 94.5%    | 0.0905 | 0.0244 |
| Multilayer Perceptron | 88.0%    | —      | —      |

---

## 🔍 Comparison of EER and T-DCF Values Across Models

| Model                  | EER     | T-DCF   |
|------------------------|---------|---------|
| **TSSD + Attention (our)** | 0.0505  | 0.0044  |
| TSSD                   | 0.0905  | 0.0244  |


---
## UI Preview

### Front Page

<img src="https://github.com/user-attachments/assets/9d034bc5-0656-4384-8b7d-fea16f7343c8" alt="Front page" width="600"/>


### Login Page

<img src="https://github.com/user-attachments/assets/a8137526-bc52-4fc5-a06e-654e4d7784f4" alt="Login Page" width="600"/>



### Real Audio Result

<img src="https://github.com/user-attachments/assets/ccdba467-42d7-44b3-997e-4eb4de6d44b9" alt="REAL Audio result" width="600"/>



### Fake Audio Result

<img src="https://github.com/user-attachments/assets/c3c8adcd-3ec5-41a5-8bb8-3698c0df43af" alt="FAKE Audio result" width="600"/>
