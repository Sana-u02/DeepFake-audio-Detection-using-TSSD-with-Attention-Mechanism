# DeepFake Audio Detection using TSSD with Attention Mechanism

As Deepfake technology advances, audio-based forgeries pose significant threats to digital security, including voice impersonation, misinformation, and fraud. This project aims to tackle these challenges by developing a robust deep learning model capable of distinguishing between real and synthetic audio.

## Project Overview

This system leverages a **Time-Series Signal Detection (TSSD)** model integrated with an **Attention Mechanism** to enhance focus on key audio features. The model is trained on benchmark datasets and achieves high accuracy in classifying real and fake audio.

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

- **Real audio clips**: 15,000 samples from public speech datasets  
- **Fake audio clips**: 15,000 clips generated using TTS models like WaveNet  
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

## 💻 Getting Started

```bash
# Clone this repo
git clone https://github.com/yourusername/deepfake-audio-detection-tssd.git
cd deepfake-audio-detection-tssd

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python app.py


