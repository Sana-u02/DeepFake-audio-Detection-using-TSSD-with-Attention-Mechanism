import torch
import librosa
import numpy as np

# Define the runtest function to accept both audio file path and model
def runtest(audio_file_path, model):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_file_path, sr=None)

    # Extract features from the audio data
    features = extract_features(audio_data, sample_rate)

    # Prepare the features for the model
    input_tensor = torch.tensor(features).unsqueeze(0).float()  # Add .float() to ensure it's a float tensor

    # Get model predictions
    with torch.no_grad():
        output = model(input_tensor)  # Pass the features to the model

    # Apply sigmoid for binary classification
    probability = torch.sigmoid(output).item()

    # Set threshold to classify as real or fake
    predicted_class = 1 if probability >= 0.5 else 0

    # Return a meaningful result based on the predicted class
    if predicted_class == 0:  # Adjust based on your class mapping
        return "Real Audio File"
    else:
        return "Fake Audio File"


# Implement feature extraction based on your specific requirements
def extract_features(audio_data, sample_rate):
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    
    # Extract Mel spectrogram features
    mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sample_rate).T, axis=0)
    
    # Extract chroma features
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).T, axis=0)
    
    # Extract zero-crossing rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio_data).T, axis=0)
    
    # Extract spectral centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).T, axis=0)
    
    # Extract spectral flatness
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio_data).T, axis=0)
    
    # Concatenate all features into a single feature vector
    feature_vector = np.hstack([mfccs, mel_spectrogram, chroma, zero_crossing_rate, spectral_centroid, spectral_flatness])
    
    # Ensure feature vector is converted to 2D if needed for the model
    feature_vector = feature_vector.reshape(1, -1)

    return feature_vector
