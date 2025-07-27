# VoiceMap - Cognitive Decline Detection

A machine learning pipeline for detecting early signs of cognitive decline (dementia) from speech recordings using deep learning and audio processing techniques.

## Overview

This project implements a CNN-based classifier that analyzes speech patterns to distinguish between healthy controls and individuals with dementia. The system extracts log-mel spectrograms from audio recordings and uses a deep convolutional neural network for classification.

## Features

- **Audio Processing**: Extracts log-mel spectrograms from various audio formats
- **Deep Learning**: CNN architecture optimized for audio classification
- **Data Augmentation**: Built-in augmentation for improved model robustness
- **Real-time Inference**: Live voice recording and analysis capabilities
- **Reproducible Training**: Deterministic training with fixed random seeds

## Project Structure

```
VoiceMapPrototype/
├── train.py                 # Main training script
├── inference.py             # Inference on audio files
├── record_and_detect.py     # Real-time voice recording and analysis
├── datasets.py              # Data loading and feature extraction
├── organize_dementia_data.py # Data organization utilities
├── requirements.txt         # Python dependencies
├── prompt.txt              # Original project specification
└── models/                 # Trained models (not in repo)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd VoiceMapPrototype
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional packages for data augmentation:**
   ```bash
   pip install audiomentations
   ```

## Usage

### Training a Model

Train on a dataset with the following structure:
```
dataset/
├── control/
│   └── [audio files]
└── dementia/
    └── [audio files]
```

```bash
python train.py --data_dir your_dataset
```

### Running Inference

Analyze an audio file:
```bash
python inference.py path/to/audio.wav
```

### Real-time Voice Analysis

Record and analyze your voice:
```bash
python record_and_detect.py
```

## Model Architecture

The system uses a deep CNN with:
- 4 convolutional layers with batch normalization and dropout
- Log-mel spectrogram input (128 mel bands × 256 time frames)
- Data augmentation (noise, pitch shift, time stretch, gain)
- Class-weighted loss for imbalanced datasets

## Data Requirements

- Audio files: `.wav`, `.mp3`, `.flac`, `.m4a`
- Sample rate: 22050 Hz (automatically resampled)
- Duration: Variable (padded/truncated to 256 frames)

## Performance

The model achieves classification accuracy on dementia detection tasks. Performance varies based on:
- Dataset size and quality
- Audio recording conditions
- Task type (e.g., Cookie Theft, Fluency)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation format here]
```

## Acknowledgments

- Built with PyTorch, librosa, and scikit-learn
- Inspired by research on speech-based dementia detection
- Uses data augmentation techniques for improved generalization 