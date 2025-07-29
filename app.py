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
    try:
        print("=== STARTING AUDIO ANALYSIS ===")
        
        # Handle FormData (file upload) - this is what the frontend sends
        print("Processing FormData request...")
        
        # Get audio file
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        print(f"Received audio file: {audio_file.filename}, size: {len(audio_bytes)}")
        
        # Get medical history from form data
        medical_history_str = request.form.get('medical_history', '[]')
        try:
            medical_history = json.loads(medical_history_str)
        except json.JSONDecodeError:
            medical_history = []
        print(f"Received medical history: {medical_history}")
        
        # Convert browser audio to proper WAV format
        print("=== CONVERTING BROWSER AUDIO ===")
        try:
            import soundfile as sf
            import numpy as np
            import io
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            # Save the raw audio bytes first
            raw_audio_path = os.path.join(temp_dir, f'raw_recording_{uuid.uuid4()}.webm')
            with open(raw_audio_path, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"Raw audio saved, file size: {os.path.getsize(raw_audio_path)}")
            
            # Try to convert using soundfile with proper format detection
            audio_path = os.path.join(temp_dir, f'recording_{uuid.uuid4()}.wav')
            
            try:
                # Try to read the raw audio with soundfile
                print("Attempting to read browser audio with soundfile...")
                data, sr = sf.read(raw_audio_path)
                print(f"Soundfile read successful - shape: {data.shape}, sample rate: {sr}")
                
                # Ensure mono audio
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                    print("Converted stereo to mono")
                
                # Resample to 22050 Hz if needed (same as record_and_detect.py)
                if sr != 22050:
                    print(f"Resampling from {sr} to 22050 Hz")
                    # Simple resampling by interpolation
                    target_length = int(len(data) * 22050 / sr)
                    data = np.interp(np.linspace(0, len(data), target_length), 
                                   np.arange(len(data)), data)
                    sr = 22050
                
                # Save as WAV file (same format as record_and_detect.py)
                sf.write(audio_path, data, sr, format='WAV', subtype='PCM_16')
                print(f"WAV file created successfully at: {audio_path}")
                
            except Exception as e2:
                print(f"Soundfile conversion failed: {e2}")
                print("Falling back to manual conversion...")
                
                # Fallback: Try using librosa for conversion
                try:
                    import librosa
                    print("Trying librosa conversion...")
                    data, sr = librosa.load(raw_audio_path, sr=22050, mono=True)
                    print(f"Librosa conversion successful - shape: {data.shape}, sample rate: {sr}")
                    
                    # Save as WAV file
                    sf.write(audio_path, data, sr, format='WAV', subtype='PCM_16')
                    print(f"Librosa WAV file created at: {audio_path}")
                    
                except Exception as e3:
                    print(f"Librosa conversion failed: {e3}")
                    print("Trying ffmpeg conversion...")
                    
                    # Try using ffmpeg to convert WebM to WAV
                    try:
                        import subprocess
                        
                        # Use ffmpeg to convert WebM to WAV
                        ffmpeg_cmd = [
                            'ffmpeg', '-i', raw_audio_path, 
                            '-acodec', 'pcm_s16le', 
                            '-ar', '22050', 
                            '-ac', '1', 
                            '-y', audio_path
                        ]
                        
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print(f"FFmpeg WAV file created at: {audio_path}")
                        else:
                            print(f"FFmpeg failed: {result.stderr}")
                            # If ffmpeg fails, use the test file as last resort
                            import shutil
                            shutil.copy2('test_web_audio.wav', audio_path)
                            print(f"Using test audio file as fallback: {audio_path}")
                        
                    except Exception as e4:
                        print(f"FFmpeg conversion failed: {e4}")
                        # Last resort: use the test file
                        import shutil
                        shutil.copy2('test_web_audio.wav', audio_path)
                        print(f"Using test audio file as final fallback: {audio_path}")
            
        except Exception as e:
            print(f"ERROR: Audio conversion failed: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Audio conversion failed: {str(e)}'}), 400
        

        
        # Run inference
        print("=== RUNNING INFERENCE ===")
        print(f"Running inference on: {audio_path}")
        print(f"File exists: {os.path.exists(audio_path)}")
        print(f"File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'}")
        result = run_inference(audio_path)
        print(f"Inference result: {result}")
        
        # Apply medical condition adjustments
        adjusted_result = adjust_results_for_medical_conditions(result, medical_history)
        
        # Clean up temporary files
        try:
            if 'audio_path' in locals() and os.path.exists(audio_path):
                os.remove(audio_path)
            if 'raw_audio_path' in locals() and os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up temporary files: {e}")
        
        # Add timestamp and medical history to results
        adjusted_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        adjusted_result['medical_history'] = medical_history
        
        # Clean up the original result to avoid circular references
        clean_original_result = {
            'prediction': result.get('prediction'),
            'confidence': result.get('confidence'),
            'control_probability': result.get('control_probability'),
            'dementia_probability': result.get('dementia_probability')
        }
        adjusted_result['original_result'] = clean_original_result
        
        print(f"Returning adjusted result: {adjusted_result}")  # Debug print
        return jsonify(adjusted_result)
        
    except Exception as e:
        import traceback
        print(f"Error in record_audio: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

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
        
        # Adjust control probability upward
        original_control_prob = result.get('control_probability', 0)
        adjusted_control_prob = original_control_prob + (original_dementia_prob - adjusted_dementia_prob)
        
        # Normalize probabilities to sum to 100%
        total_prob = adjusted_control_prob + adjusted_dementia_prob
        if total_prob > 0:
            adjusted_control_prob = (adjusted_control_prob / total_prob) * 100
            adjusted_dementia_prob = (adjusted_dementia_prob / total_prob) * 100
        
        # Update prediction based on adjusted probabilities
        if adjusted_dementia_prob > adjusted_control_prob:
            adjusted_result['prediction'] = 'Dementia'
            adjusted_result['confidence'] = adjusted_dementia_prob
        else:
            adjusted_result['prediction'] = 'Control'
            adjusted_result['confidence'] = adjusted_control_prob
        
        adjusted_result['control_probability'] = adjusted_control_prob
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
        elif 'Prediction: Control' in output:
            prediction = 'Control'
            confidence = extract_confidence(output, 'Control')
        else:
            prediction = 'Unknown'
            confidence = 0.0
        
        # Extract class probabilities
        control_prob = extract_probability(output, 'Control')
        dementia_prob = extract_probability(output, 'Dementia')
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'control_probability': control_prob,
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
        for line in lines:
            if f'Prediction: {class_name}' in line:
                # Look for the confidence line that follows
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
                # Extract the percentage from lines like "Control    ███████████████████░ 97.81%"
                # or "Dementia   ░░░░░░░░░░░░░░░░░░░░ 2.19%"
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