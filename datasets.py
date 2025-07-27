import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

def extract_features(audio_path: str, sr: float = 22050) -> np.ndarray:
    """
    Extract audio features from a single audio file.
    
    Args:
        audio_path: Path to the audio file
        sr: Sample rate for loading audio
    
    Returns:
        Feature vector as numpy array
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCCs (13 dimensions)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Delta and Delta-Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
        mfcc_delta_std = np.std(mfcc_delta, axis=1)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)
        mfcc_delta2_std = np.std(mfcc_delta2, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_std = np.std(bandwidth)
        
        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Extract Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # Extract Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean, mfcc_std,  # 26
            mfcc_delta_mean, mfcc_delta_std,  # 26
            mfcc_delta2_mean, mfcc_delta2_std,  # 26
            chroma_mean, chroma_std,  # 24
            [bandwidth_mean, bandwidth_std],  # 2
            [rolloff_mean, rolloff_std],  # 2
            [rms_mean, rms_std],  # 2
            [zcr_mean, zcr_std],  # 2
            [centroid_mean, centroid_std]  # 2
        ])
        return features
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        # Return zero vector if processing fails
        return np.zeros(110)

def extract_logmel(audio_path: str, sr: float = 22050, n_mels: int = 128, n_frames: int = 256, augment: bool = False) -> np.ndarray:
    """
    Extract a log-mel spectrogram from an audio file, padded or truncated to (n_mels, n_frames).
    If augment is True, apply data augmentation to the waveform.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        if augment:
            try:
                from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Gain, Shift
                augmenter = Compose([
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                    PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                    Gain(min_gain_db=-6, max_gain_db=6, p=0.3),
                    Shift(min_shift=-0.2, max_shift=0.2, p=0.3)
                ])
                y = augmenter(samples=y, sample_rate=int(sr))
            except ImportError:
                print("audiomentations not installed, skipping augmentation.")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        logmel = librosa.power_to_db(mel)
        # Pad or truncate to n_frames
        if logmel.shape[1] < n_frames:
            pad_width = n_frames - logmel.shape[1]
            logmel = np.pad(logmel, ((0,0),(0,pad_width)), mode='constant')
        else:
            logmel = logmel[:, :n_frames]
        return logmel.astype(np.float32)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return np.zeros((n_mels, n_frames), dtype=np.float32)

def load_dataset(root_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from directory structure and extract features.
    
    Args:
        root_dir: Root directory containing control/, dementia/, mci/ folders
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (n_samples,)
    """
    features_list = []
    labels_list = []
    label_mapping = {'control': 0, 'dementia': 1, 'mci': 2}
    
    # Supported audio formats
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    
    print("Loading dataset...")
    
    for label_name, label_id in label_mapping.items():
        label_dir = os.path.join(root_dir, label_name)
        
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} not found")
            continue
            
        print(f"Processing {label_name} samples...")
        
        # Recursively find all audio files
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_path = os.path.join(root, file)
                    
                    # Extract features
                    features = extract_features(audio_path)
                    
                    # Skip if features are all zeros (processing failed)
                    if not np.all(features == 0):
                        features_list.append(features)
                        labels_list.append(label_id)
    
    if not features_list:
        raise ValueError("No valid audio files found in the dataset directory")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: {np.bincount(y)}")
    
    return X, y

def load_logmel_dataset(root_dir: str, n_mels: int = 128, n_frames: int = 256, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset and return log-mel spectrograms and labels for CNN input.
    If augment is True, apply augmentation to each sample.
    """
    features_list = []
    labels_list = []
    label_mapping = {'control': 0, 'dementia': 1}
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
    print("Loading log-mel spectrogram dataset... (augment={})".format(augment))
    for label_name, label_id in label_mapping.items():
        label_dir = os.path.join(root_dir, label_name)
        if not os.path.exists(label_dir):
            print(f"Warning: Directory {label_dir} not found")
            continue
        print(f"Processing {label_name} samples...")
        for root, dirs, files in os.walk(label_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_path = os.path.join(root, file)
                    logmel = extract_logmel(audio_path, n_mels=n_mels, n_frames=n_frames, augment=augment)
                    if not np.all(logmel == 0):
                        features_list.append(logmel)
                        labels_list.append(label_id)
    if not features_list:
        raise ValueError("No valid audio files found in the dataset directory")
    X = np.stack(features_list)  # (n_samples, n_mels, n_frames)
    y = np.array(labels_list)
    print(f"Log-mel dataset loaded: {X.shape[0]} samples, shape per sample: {X.shape[1:]} (label dist: {np.bincount(y)})")
    return X, y

def save_scaler(scaler: StandardScaler, filepath: str):
    """Save fitted scaler to file."""
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")

def load_scaler(filepath: str) -> StandardScaler:
    """Load fitted scaler from file."""
    return joblib.load(filepath)

def get_feature_names() -> List[str]:
    """Get names of extracted features."""
    feature_names = []
    # MFCC features
    for i in range(13):
        feature_names.append(f"mfcc_{i+1}_mean")
    for i in range(13):
        feature_names.append(f"mfcc_{i+1}_std")
    # Delta MFCC
    for i in range(13):
        feature_names.append(f"mfcc_delta_{i+1}_mean")
    for i in range(13):
        feature_names.append(f"mfcc_delta_{i+1}_std")
    # Delta-Delta MFCC
    for i in range(13):
        feature_names.append(f"mfcc_delta2_{i+1}_mean")
    for i in range(13):
        feature_names.append(f"mfcc_delta2_{i+1}_std")
    # Chroma
    for i in range(12):
        feature_names.append(f"chroma_{i+1}_mean")
    for i in range(12):
        feature_names.append(f"chroma_{i+1}_std")
    # Bandwidth
    feature_names.extend(["bandwidth_mean", "bandwidth_std"])
    # Rolloff
    feature_names.extend(["rolloff_mean", "rolloff_std"])
    # Energy features
    feature_names.extend(["rms_mean", "rms_std"])
    # Zero crossing rate features
    feature_names.extend(["zcr_mean", "zcr_std"])
    # Spectral centroid features
    feature_names.extend(["spectral_centroid_mean", "spectral_centroid_std"])
    return feature_names 