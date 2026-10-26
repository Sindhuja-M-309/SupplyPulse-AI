import numpy as np
import scipy.io.wavfile as wav
from scipy.fft import fft
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_features(file_path):
    # 1. Load audio using SciPy / SoundFile (No Librosa/Numba required)
    sr, y = wav.read(file_path)
    
    # Normalize if integer array
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / 32768.0
        
    # Convert stereo to mono if needed
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
        
    # 2. Extract Spectral Centroid manually via FFT
    magnitude = np.abs(fft(y))
    freqs = np.fft.fftfreq(len(magnitude), 1.0/sr)
    
    # Keep positive frequencies only
    pos_mask = freqs >= 0
    freqs = freqs[pos_mask]
    magnitude = magnitude[pos_mask]
    
    spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-10)
    
    # 3. Simple time-domain statistics (energy & zero-crossing rate)
    energy = np.sum(y**2) / len(y)
    zcr = np.sum(np.diff(np.sign(y)) != 0) / len(y)
    
    return np.array([spectral_centroid, energy, zcr])

def train_audio_model():
    X, y_labels = [], []
    
    firm_dir = './data/audio/firm'
    soft_dir = './data/audio/soft'
    
    if os.path.exists(firm_dir):
        for f in os.listdir(firm_dir):
            if f.endswith('.wav'):
                X.append(extract_features(os.path.join(firm_dir, f)))
                y_labels.append(1) # Firm
                
    if os.path.exists(soft_dir):
        for f in os.listdir(soft_dir):
            if f.endswith('.wav'):
                X.append(extract_features(os.path.join(soft_dir, f)))
                y_labels.append(0) # Soft
                
    if len(X) < 2:
        print("⚠️ Place audio files in ./data/audio/firm and ./data/audio/soft first!")
        return

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y_labels)
    joblib.dump(model, 'audio_firmness_model.pkl')
    print("✅ Audio model trained successfully without Librosa/Numba errors!")

if __name__ == "__main__":
    train_audio_model()