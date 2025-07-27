import sys
import os
import numpy as np
import torch
import torch.nn as nn
from datasets import extract_logmel

def load_cnn_model(model_path, n_mels=128, n_frames=256, n_classes=2, device=None):
    if device is None:
        device = torch.device("cpu")
    class DeeperAudioCNN(nn.Module):
        def __init__(self, n_mels=128, n_frames=256, n_classes=2):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.3)
            )
            conv_out_size = (n_mels // 16) * (n_frames // 16) * 256
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, n_classes)
            )
        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    model = DeeperAudioCNN(n_mels=n_mels, n_frames=n_frames, n_classes=n_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_audio(audio_path: str, model_path: str = "models/cnn_model.pt", n_mels: int = 128, n_frames: int = 256) -> dict:
    class_names = ['Control', 'Dementia']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Extract log-mel spectrogram
        logmel = extract_logmel(audio_path, n_mels=n_mels, n_frames=n_frames)
        if np.all(logmel == 0):
            raise ValueError("Feature extraction failed - audio file may be corrupted or unsupported")
        features = logmel[None, None, :, :]  # (1, 1, n_mels, n_frames)
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        # Load model
        model = load_cnn_model(model_path, n_mels=n_mels, n_frames=n_frames, n_classes=len(class_names), device=device)
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            prediction = int(np.argmax(probabilities))
            confidence = float(np.max(probabilities))
        result = {
            'audio_file': os.path.basename(audio_path),
            'predicted_class': class_names[prediction],
            'predicted_class_id': prediction,
            'confidence': confidence,
            'class_probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        return result
    except Exception as e:
        return {
            'error': str(e),
            'audio_file': os.path.basename(audio_path) if os.path.exists(audio_path) else audio_path
        }

def print_prediction_result(result: dict):
    if 'error' in result:
        print(f"\n❌ ERROR: {result['error']}")
        return
    print("\n" + "="*60)
    print("VOICEMAP - COGNITIVE DECLINE DETECTION")
    print("="*60)
    print(f"Audio File: {result['audio_file']}")
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print("\nClass Probabilities:")
    for class_name, prob in result['class_probabilities'].items():
        bar_length = int(prob * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {class_name:<10} {bar} {prob:.2%}")
    print("\nInterpretation:")
    if result['predicted_class'] == 'Control':
        print("  ✅ No significant cognitive impairment detected")
    else:
        print("  🚨 Dementia indicators detected - recommend clinical evaluation")
    if result['confidence'] < 0.6:
        print("  ⚠️  Low confidence prediction - consider retesting")
    print("="*60)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <audio_file_path>")
        print("Example: python inference.py patient_audio.wav")
        sys.exit(1)
    audio_path = sys.argv[1]
    print("VoiceMap Inference - Cognitive Decline Detection")
    print("="*50)
    model_path = "models/cnn_model.pt"
    if not os.path.exists(model_path):
        print("❌ No trained CNN model found!")
        print("Please run train.py first to train a model.")
        sys.exit(1)
    result = predict_audio(audio_path, model_path)
    print_prediction_result(result)
    if 'error' in result:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 