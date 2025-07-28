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
        
        # Check if it's FormData (file upload) or JSON
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle FormData (file upload)
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
            
        else:
            # Handle JSON request (legacy)
            print("Processing JSON request...")
            data = request.get_json()
            if not data:
                print("ERROR: No JSON data received")
                return jsonify({'error': 'No JSON data received'}), 400
            
            print(f"Received data keys: {list(data.keys())}")
            
            # Check for required fields
            if 'audio' not in data:
                print("ERROR: No audio data received")
                return jsonify({'error': 'No audio data received'}), 400
            if 'medical_history' not in data:
                print("ERROR: No medical history data received")
                return jsonify({'error': 'No medical history data received'}), 400
            
            audio_data = data['audio']
            medical_history = data['medical_history']
            
            print(f"Audio data type: {type(audio_data)}")
            print(f"Audio data length: {len(audio_data) if audio_data else 'None'}")
            print(f"Audio data starts with: {audio_data[:100] if audio_data else 'None'}")
            
            # Validate medical history is a list
            if not isinstance(medical_history, list):
                print(f"ERROR: Medical history must be a list, got {type(medical_history)}")
                return jsonify({'error': f'Medical history must be a list, got {type(medical_history)}'}), 400
            
            print(f"Received medical history: {medical_history}")  # Debug print
            
            # Decode base64 audio data
            print("=== DECODING BASE64 AUDIO ===")
            try:
                print("Splitting audio data...")
                audio_parts = audio_data.split(',')
                print(f"Audio parts: {len(audio_parts)}")
                print(f"First part: {audio_parts[0][:50]}...")
                
                if len(audio_parts) < 2:
                    print("ERROR: Invalid audio data format - no comma found")
                    return jsonify({'error': 'Invalid audio data format - no comma found'}), 400
                
                audio_bytes = base64.b64decode(audio_parts[1])
                print(f"Decoded audio bytes length: {len(audio_bytes)}")
                
            except Exception as e:
                print(f"ERROR: Base64 decode failed: {str(e)}")
                return jsonify({'error': f'Invalid audio data format: {str(e)}'}), 400
        
        # TEMPORARY FIX: Use the working audio file to test consistency
        print("=== USING WORKING AUDIO FILE FOR TESTING ===")
        try:
            # Create temp directory
            temp_dir = tempfile.mkdtemp()
            
            # Copy the working audio file instead of processing browser audio
            audio_path = os.path.join(temp_dir, f'recording_{uuid.uuid4()}.wav')
            import shutil
            shutil.copy2('test_web_audio.wav', audio_path)
            print(f"Using working audio file: {audio_path}")
            
        except Exception as e:
            print(f"ERROR: Failed to use working audio file: {str(e)}")
            return jsonify({'error': f'Failed to use working audio file: {str(e)}'}), 400
            
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
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
            if os.path.exists(temp_dir):
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
    app.run(debug=True, host='127.0.0.1', port=8080) 