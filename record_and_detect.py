import sounddevice as sd
from scipy.io.wavfile import write
import os
import sys
import subprocess

def record_audio(filename='recorded.wav', duration=5, fs=22050):
    print(f"Recording {duration} seconds of audio. Please speak into the microphone...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, audio)
    print(f"Audio saved to {filename}")

def run_inference(audio_path):
    print("Running inference...")
    result = subprocess.run([sys.executable, 'inference.py', audio_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main():
    duration = input("Enter recording duration in seconds (default 5): ")
    try:
        duration = float(duration)
    except Exception:
        duration = 5
    record_audio('recorded.wav', duration=int(duration))
    run_inference('recorded.wav')

if __name__ == '__main__':
    main() 