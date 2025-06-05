import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tssd import TSSD  # Import the TSSD model from tssd1.py

data_dir = "data"

# Function to load audio files and extract features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract MFCC features
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        
        # Extract Mel spectrogram features
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
        
        # Extract chroma features
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
        
        # Extract zero-crossing rate
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        
        # Extract spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).T, axis=0)
        
        # Extract spectral flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio).T, axis=0)
        
        # Concatenate all features into a single feature vector
        feature_vector = np.hstack([mfccs, mel_spectrogram, chroma, zero_crossing_rate, spectral_centroid, spectral_flatness])
        
        return feature_vector
    except Exception as e:
        print("Error encountered while parsing file:", file_path)
        return None

# Load data
def load_data(data_dir):
    fake_files = [os.path.join(data_dir, "fake", f) for f in os.listdir(os.path.join(data_dir, "fake")) if f.endswith(".wav")]
    real_files = [os.path.join(data_dir, "real", f) for f in os.listdir(os.path.join(data_dir, "real")) if f.endswith(".wav")]

    fake_labels = [0] * len(fake_files)  # 0 for fake
    real_labels = [1] * len(real_files)  # 1 for real

    files = fake_files + real_files
    labels = fake_labels + real_labels

    return files, labels

files, labels = load_data(data_dir)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(files, labels, test_size=0.2, random_state=42)

# Extract features for training and testing
X_train = [extract_features(file) for file in X_train]
X_test = [extract_features(file) for file in X_test]

# Filter out any None values in features and corresponding labels
X_train, y_train = zip(*[(x, y) for x, y in zip(X_train, y_train) if x is not None])
X_test, y_test = zip(*[(x, y) for x, y in zip(X_test, y_test) if x is not None])

# Convert lists to numpy arrays
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the TSSD model
model = TSSD(in_dim=X_train.shape[1])

# Set up the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the TSSD model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Move tensors to the appropriate device
        inputs, labels = inputs, labels.view(-1, 1)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * inputs.size(0)

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Save the trained model
model_save_path = "C:\\Users\\dell\\Desktop\\deepfake audio detection\\tssd1_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluating the TSSD model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs, labels.view(-1, 1)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Evaluating training accuracy
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs, labels.view(-1, 1)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total
    print(f"Training Accuracy: {train_accuracy:.2f}%")
