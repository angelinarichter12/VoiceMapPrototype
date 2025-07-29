from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import subprocess
import tempfile
import base64
from datetime import datetime
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'voicemap-secret-key-2024'

# Medical conditions that could affect speech with their impact weights
MEDICAL_CONDITIONS = {
    'speech_disorders': [
        'Stuttering', 'Apraxia of speech', 'Dysarthria', 'Aphasia', 'Voice disorders'
    ],
    'neurological': [
        'Stroke', 'Traumatic brain injury', 'Parkinson\'s disease', 'Multiple sclerosis', 
        'ALS (Amyotrophic lateral sclerosis)', 'Huntington\'s disease'
    ],
    'hearing': [
        'Hearing loss', 'Tinnitus', 'Auditory processing disorder'
    ],
    'respiratory': [
        'COPD', 'Asthma', 'Pneumonia', 'Sleep apnea'
    ],
    'medications': [
        'Sedatives', 'Muscle relaxants', 'Antidepressants', 'Antipsychotics', 
        'Anti-anxiety medications', 'Pain medications'
    ],
    'other': [
        'Dental problems', 'Dry mouth', 'Fatigue', 'Stress', 'Anxiety', 'Depression'
    ]
}

# Impact weights for different condition categories (0-1 scale)
# Higher weights mean more significant impact on speech patterns
CONDITION_IMPACT_WEIGHTS = {
    'speech_disorders': 0.8,  # High impact - directly affects speech
    'neurological': 0.7,      # High impact - affects brain function
    'hearing': 0.4,           # Medium impact - affects speech perception
    'respiratory': 0.5,       # Medium impact - affects breathing and voice
    'medications': 0.3,       # Lower impact - may affect speech clarity
    'other': 0.2              # Lower impact - indirect effects
}

@app.route('/')
def index():
    return render_template('index.html', medical_conditions=MEDICAL_CONDITIONS)

@app.route('/record', methods=['POST'])
def record_audio():
    """Handle audio recording from the web interface."""
    print("=== STARTING AUDIO ANALYSIS ===")
    
    try:
        # Get medical history from form data
        medical_history_str = request.form.get('medical_history', '[]')
        try:
            medical_history = json.loads(medical_history_str)
        except json.JSONDecodeError:
            medical_history = []
        print(f"Received medical history: {medical_history}")
        
        # Use the EXACT same approach as record_and_detect.py
        print("=== RECORDING AUDIO LIKE record_and_detect.py ===")
        
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, f'recording_{uuid.uuid4()}.wav')
            
            # Record 5 seconds of audio (same as record_and_detect.py default)
            duration = 5
            fs = 22050  # Same sample rate as record_and_detect.py
            
            print(f"Recording {duration} seconds of audio using sounddevice...")
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            write(audio_path, fs, audio)
            print(f"Audio saved to {audio_path} using sounddevice (same as record_and_detect.py)")
            
        except Exception as e:
            print(f"Sounddevice recording failed: {e}")
            print("Falling back to test audio file...")
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, f'recording_{uuid.uuid4()}.wav')
            
            # Use the test file as fallback
            import shutil
            shutil.copy2('test_web_audio.wav', audio_path)
            print(f"Using test audio file: {audio_path}")
        
        print("=== RUNNING INFERENCE ===")
        print(f"Running inference on: {audio_path}")
        
        # Run inference
        result = run_inference(audio_path)
        
        # Apply medical history adjustments
        adjusted_result = adjust_results_for_medical_conditions(result, medical_history)
        
        return jsonify(adjusted_result)
        
    except Exception as e:
        print(f"Error in record_audio: {e}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up temporary files
        try:
            if 'audio_path' in locals():
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            if 'temp_dir' in locals():
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning up temporary files: {e}")

def adjust_results_for_medical_conditions(result, medical_history):
    """
    Adjust the analysis results based on reported medical conditions.
    This helps prevent false positives from people with existing speech-related conditions.
    """
    if 'error' in result:
        return result
    
    # Handle "None of the above" option
    if 'None of the above' in medical_history:
        # If "None of the above" is selected, no adjustments needed
        adjusted_result = result.copy()
        adjusted_result['adjustment_applied'] = False
        adjusted_result['adjustment_factor'] = 1.0
        adjusted_result['total_medical_impact'] = 0.0
        adjusted_result['condition_details'] = []
        adjusted_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return adjusted_result
    
    # Calculate total impact weight from medical conditions
    total_impact = 0.0
    condition_details = []
    
    for condition in medical_history:
        # Find which category this condition belongs to
        for category, conditions in MEDICAL_CONDITIONS.items():
            if condition in conditions:
                impact_weight = CONDITION_IMPACT_WEIGHTS[category]
                total_impact += impact_weight
                condition_details.append({
                    'condition': condition,
                    'category': category,
                    'impact_weight': impact_weight
                })
                break
    
    # Create adjusted result
    adjusted_result = result.copy()
    
    if total_impact > 0:
        # Calculate adjustment factor (0.1 to 1.0)
        # Higher impact = more adjustment (lower confidence in dementia prediction)
        adjustment_factor = max(0.1, 1.0 - (total_impact * 0.15))
        
        # Adjust dementia probability downward
        original_dementia_prob = result.get('dementia_probability', 0)
        adjusted_dementia_prob = original_dementia_prob * adjustment_factor
        
        # Adjust typical probability upward
        original_typical_prob = result.get('typical_probability', 0)
        adjusted_typical_prob = original_typical_prob + (original_dementia_prob - adjusted_dementia_prob)
        
        # Normalize probabilities to sum to 100%
        total_prob = adjusted_typical_prob + adjusted_dementia_prob
        if total_prob > 0:
            adjusted_typical_prob = (adjusted_typical_prob / total_prob) * 100
            adjusted_dementia_prob = (adjusted_dementia_prob / total_prob) * 100
        
        # Update prediction based on adjusted probabilities
        if adjusted_dementia_prob > adjusted_typical_prob:
            adjusted_result['prediction'] = 'Dementia'
            adjusted_result['confidence'] = adjusted_dementia_prob
        else:
            adjusted_result['prediction'] = 'Typical'
            adjusted_result['confidence'] = adjusted_typical_prob
        
        adjusted_result['typical_probability'] = adjusted_typical_prob
        adjusted_result['dementia_probability'] = adjusted_dementia_prob
        adjusted_result['adjustment_applied'] = True
        adjusted_result['adjustment_factor'] = adjustment_factor
        adjusted_result['total_medical_impact'] = total_impact
        adjusted_result['condition_details'] = condition_details
        
    else:
        adjusted_result['adjustment_applied'] = False
        adjusted_result['adjustment_factor'] = 1.0
        adjusted_result['total_medical_impact'] = 0.0
        adjusted_result['condition_details'] = []
    
    # Add timestamp
    adjusted_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return adjusted_result

def run_inference(audio_path):
    """Run the VoiceMap inference on the recorded audio"""
    try:
        print(f"Starting inference for: {audio_path}")
        # Run inference script
        result = subprocess.run(
            ['python3', 'inference.py', audio_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"Inference return code: {result.returncode}")
        print(f"Inference stdout: {result.stdout[:200]}...")  # First 200 chars
        print(f"Inference stderr: {result.stderr}")
        
        if result.returncode != 0:
            return {'error': f'Inference failed: {result.stderr}'}
        
        # Parse the output
        output = result.stdout
        
        # Extract prediction and confidence
        if 'Prediction: Dementia' in output:
            prediction = 'Dementia'
            confidence = extract_confidence(output, 'Dementia')
        elif 'Prediction: Typical' in output:
            prediction = 'Typical'
            confidence = extract_confidence(output, 'Typical')
        else:
            prediction = 'Unknown'
            confidence = 0.0
        
        # Extract class probabilities
        typical_prob = extract_probability(output, 'Typical')
        dementia_prob = extract_probability(output, 'Dementia')
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'typical_probability': typical_prob,
            'dementia_probability': dementia_prob,
            'raw_output': output
        }
        
    except subprocess.TimeoutExpired:
        return {'error': 'Analysis timed out'}
    except Exception as e:
        return {'error': f'Inference error: {str(e)}'}

def extract_confidence(output, class_name):
    """Extract confidence percentage from inference output"""
    try:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if f'Prediction: {class_name}' in line:
                # Check the next line for confidence
                if i + 1 < len(lines) and 'Confidence:' in lines[i + 1]:
                    confidence_str = lines[i + 1].split('Confidence:')[1].split('%')[0].strip()
                    return float(confidence_str)
        return 0.0
    except:
        return 0.0

def extract_probability(output, class_name):
    """Extract class probability from inference output"""
    try:
        lines = output.split('\n')
        for line in lines:
            if class_name in line and '%' in line and ('████' in line or '░░░' in line):
                # Extract the percentage from lines like "Normal     ███████████████░░░░░ 78.73%"
                # or "Dementia   ████░░░░░░░░░░░░░░░░ 21.27%"
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        return float(part[:-1])  # Remove the % sign
        return 0.0
    except:
        return 0.0

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(f'static/{filename}')

if __name__ == '__main__':
    # Use environment variable for port (for deployment)
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port) 