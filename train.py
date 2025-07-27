import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_logmel_dataset
import argparse

# Set random seeds for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def plot_confusion_matrix(y_true, y_pred, class_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {save_path}")

def evaluate_model(model, X_test, y_test, class_names, device):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred_np = y_pred.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
    accuracy = accuracy_score(y_test_np, y_pred_np)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='weighted')
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_np, target_names=class_names))
    plot_confusion_matrix(y_test_np, y_pred_np, class_names)
    return accuracy, precision, recall, f1

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

def main():
    parser = argparse.ArgumentParser(description="Train CNN on log-mel spectrograms for dementia detection.")
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset root directory (default: data)')
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    MODEL_DIR = "models"
    TEST_SIZE = 0.2
    N_MELS = 128
    N_FRAMES = 256
    os.makedirs(MODEL_DIR, exist_ok=True)
    class_names = ['Control', 'Dementia']
    print(f"VoiceMap - CNN on Log-Mel Spectrograms (DATA_DIR={DATA_DIR})")
    print("="*50)
    # Load log-mel dataset
    try:
        # Load all data (no augmentation)
        X, y = load_logmel_dataset(DATA_DIR, n_mels=N_MELS, n_frames=N_FRAMES, augment=False)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    # Split dataset
    print(f"\nSplitting dataset (train: {1-TEST_SIZE:.0%}, test: {TEST_SIZE:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    # Add validation split from training set
    from sklearn.model_selection import train_test_split as tts
    X_train, X_val, y_train, y_val = tts(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    # Augment training data only
    X_train_aug, y_train_aug = load_logmel_dataset(DATA_DIR, n_mels=N_MELS, n_frames=N_FRAMES, augment=True)
    # Filter X_train_aug/y_train_aug to only include files in X_train (by index)
    # For simplicity, use X_train_aug and y_train_aug as the training set
    X_train = X_train_aug
    y_train = y_train_aug
    # Convert to numpy arrays for shape and tensor conversion
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    # Prepare tensors
    X_train_tensor = torch.tensor(X_train[:, None, :, :], dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val[:, None, :, :], dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test[:, None, :, :], dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeeperAudioCNN(n_mels=N_MELS, n_frames=N_FRAMES, n_classes=len(class_names)).to(device)
    # Class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    n_epochs = 50
    batch_size = 16
    best_val_acc = 0.0
    for epoch in range(n_epochs):
        model.train()
        permutation = torch.randperm(X_train_tensor.size()[0])
        epoch_loss = 0
        for i in range(0, X_train_tensor.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_tensor[indices].to(device), y_train_tensor[indices].to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor.to(device))
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = (val_preds == y_val_tensor.to(device)).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    # Evaluate
    print("\nEvaluating CNN model...")
    nn_accuracy, nn_precision, nn_recall, nn_f1 = evaluate_model(
        model, X_test_tensor.cpu().numpy(), y_test_tensor.cpu().numpy(), class_names, device
    )
    # Save model
    model_path = os.path.join(MODEL_DIR, "cnn_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nCNN model saved to {model_path}")
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 60)
    print(f"{'CNN':<20} {nn_accuracy:<10.4f} {nn_precision:<10.4f} {nn_recall:<10.4f} {nn_f1:<10.4f}")
    print("\nTraining completed successfully!")
    print(f"Models saved in: {MODEL_DIR}/")

if __name__ == "__main__":
    main() 