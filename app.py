from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for
import os
import json
import subprocess
import tempfile
import base64
from datetime import datetime, timedelta
import uuid
import numpy as np
import pickle
from collections import defaultdict
import librosa
import warnings
warnings.filterwarnings('ignore')

# Import Supabase integration
from user_management import user_manager
from supabase_config import is_supabase_available

app = Flask(__name__)
app.config['SECRET_KEY'] = 'voicemap-secret-key-2024'

# Check if Supabase is available
SUPABASE_AVAILABLE = is_supabase_available()

# Longitudinal tracking data structure
LONGITUDINAL_DATA_FILE = 'longitudinal_data.pkl'

# Emotion and sentiment analysis features
EMOTION_FEATURES = {
    'valence': 0.0,  # Positive/negative emotion (-1 to 1)
    'arousal': 0.0,  # Energy level (0 to 1)
    'flat_affect': 0.0,  # Monotony in speech (0 to 1)
    'speech_rate': 0.0,  # Words per minute
    'pause_frequency': 0.0,  # Frequency of pauses
    'voice_tremor': 0.0,  # Voice stability
    'breathing_pattern': 0.0  # Breathing regularity
}

# Explainable AI feedback categories
EXPLAINABLE_FEATURES = {
    'speech_clarity': {'score': 0.0, 'feedback': ''},
    'fluency': {'score': 0.0, 'feedback': ''},
    'prosody': {'score': 0.0, 'feedback': ''},
    'articulation': {'score': 0.0, 'feedback': ''},
    'voice_quality': {'score': 0.0, 'feedback': ''},
    'cognitive_load': {'score': 0.0, 'feedback': ''}
}

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

def load_longitudinal_data():
    """Load longitudinal tracking data from file."""
    if os.path.exists(LONGITUDINAL_DATA_FILE):
        try:
            with open(LONGITUDINAL_DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                # Ensure it's a defaultdict
                if isinstance(data, defaultdict):
                    return data
                else:
                    # Convert to defaultdict if it's not
                    return defaultdict(list, data)
        except Exception as e:
            print(f"Error loading longitudinal data: {e}")
    return defaultdict(list)

def save_longitudinal_data(data):
    """Save longitudinal tracking data to file."""
    try:
        with open(LONGITUDINAL_DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving longitudinal data: {e}")

def analyze_emotion_and_sentiment(audio_path):
    """Analyze emotion and sentiment from audio using acoustic features."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Extract emotion-related features
        features = {}
        
        # Speech rate (approximate)
        duration = len(y) / sr
        features['speech_rate'] = min(1.0, duration / 5.0)  # Normalize to 5 seconds
        
        # Voice tremor (jitter)
        if len(y) > sr:  # Need at least 1 second
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
            pitch_values = pitches[magnitudes > 0.1]
            
            if len(pitch_values) > 10:
                # Calculate jitter (voice tremor)
                pitch_diff = np.diff(pitch_values)
                features['voice_tremor'] = min(1.0, np.std(pitch_diff) / 100.0)
            else:
                features['voice_tremor'] = 0.0
        else:
            features['voice_tremor'] = 0.0
        
        # Flat affect (monotony) - measure pitch variation
        if len(y) > sr:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > 0.1]
            if len(pitch_values) > 10:
                pitch_std = np.std(pitch_values)
                features['flat_affect'] = max(0.0, 1.0 - (pitch_std / 200.0))  # Normalize
            else:
                features['flat_affect'] = 0.5
        else:
            features['flat_affect'] = 0.5
        
        # Pause frequency
        rms = librosa.feature.rms(y=y)
        silence_threshold = np.percentile(rms, 20)
        silence_frames = np.sum(rms < silence_threshold)
        features['pause_frequency'] = min(1.0, silence_frames / len(rms[0]))
        
        # Breathing pattern (spectral centroid variation)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['breathing_pattern'] = min(1.0, np.std(spectral_centroids) / 1000.0)
        
        # Valence and arousal (simplified)
        # Higher energy = higher arousal
        energy = np.mean(librosa.feature.rms(y=y))
        features['arousal'] = min(1.0, energy / 0.1)
        
        # Valence based on spectral features (simplified)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['valence'] = max(-1.0, min(1.0, (np.mean(spectral_rolloff) - 2000) / 2000))
        
        return features
    except Exception as e:
        print(f"Error in emotion analysis: {e}")
        return {
            'valence': 0.0,
            'arousal': 0.0,
            'flat_affect': 0.0,
            'speech_rate': 0.0,
            'pause_frequency': 0.0,
            'voice_tremor': 0.0,
            'breathing_pattern': 0.0
        }

def generate_explainable_feedback(audio_path, prediction, confidence):
    """Generate explainable AI feedback with specific voice segment analysis."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        
        feedback = {}
        
        # Speech clarity (based on spectral features)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        clarity_score = min(1.0, np.mean(spectral_centroids) / 2000.0)
        feedback['speech_clarity'] = {
            'score': clarity_score,
            'feedback': f"Speech clarity is {'excellent' if clarity_score > 0.8 else 'good' if clarity_score > 0.6 else 'fair' if clarity_score > 0.4 else 'needs improvement'}"
        }
        
        # Fluency (based on pause patterns)
        rms = librosa.feature.rms(y=y)
        silence_threshold = np.percentile(rms, 20)
        silence_frames = np.sum(rms < silence_threshold)
        fluency_score = max(0.0, 1.0 - (silence_frames / len(rms[0])))
        feedback['fluency'] = {
            'score': fluency_score,
            'feedback': f"Speech fluency is {'smooth' if fluency_score > 0.8 else 'moderate' if fluency_score > 0.6 else 'has some interruptions' if fluency_score > 0.4 else 'fragmented'}"
        }
        
        # Prosody (intonation patterns)
        if len(y) > sr:
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[magnitudes > 0.1]
            if len(pitch_values) > 10:
                pitch_variation = np.std(pitch_values)
                prosody_score = min(1.0, pitch_variation / 200.0)
            else:
                prosody_score = 0.5
        else:
            prosody_score = 0.5
        
        feedback['prosody'] = {
            'score': prosody_score,
            'feedback': f"Intonation patterns are {'varied and natural' if prosody_score > 0.8 else 'moderately varied' if prosody_score > 0.6 else 'somewhat flat' if prosody_score > 0.4 else 'monotone'}"
        }
        
        # Articulation (based on MFCC features)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_variance = np.var(mfccs, axis=1)
        articulation_score = min(1.0, np.mean(mfcc_variance) / 10.0)
        feedback['articulation'] = {
            'score': articulation_score,
            'feedback': f"Articulation is {'clear and precise' if articulation_score > 0.8 else 'generally clear' if articulation_score > 0.6 else 'somewhat unclear' if articulation_score > 0.4 else 'unclear'}"
        }
        
        # Voice quality
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        voice_quality_score = min(1.0, np.mean(spectral_rolloff) / 3000.0)
        feedback['voice_quality'] = {
            'score': voice_quality_score,
            'feedback': f"Voice quality is {'excellent' if voice_quality_score > 0.8 else 'good' if voice_quality_score > 0.6 else 'fair' if voice_quality_score > 0.4 else 'poor'}"
        }
        
        # Cognitive load (based on speech rate and pauses)
        cognitive_load_score = 1.0 - (fluency_score * 0.7 + clarity_score * 0.3)
        feedback['cognitive_load'] = {
            'score': cognitive_load_score,
            'feedback': f"Cognitive load appears {'low' if cognitive_load_score < 0.3 else 'moderate' if cognitive_load_score < 0.6 else 'high' if cognitive_load_score < 0.8 else 'very high'}"
        }
        
        return feedback
    except Exception as e:
        print(f"Error in explainable feedback: {e}")
        return {
            'speech_clarity': {'score': 0.5, 'feedback': 'Analysis unavailable'},
            'fluency': {'score': 0.5, 'feedback': 'Analysis unavailable'},
            'prosody': {'score': 0.5, 'feedback': 'Analysis unavailable'},
            'articulation': {'score': 0.5, 'feedback': 'Analysis unavailable'},
            'voice_quality': {'score': 0.5, 'feedback': 'Analysis unavailable'},
            'cognitive_load': {'score': 0.5, 'feedback': 'Analysis unavailable'}
        }

def analyze_longitudinal_trends(user_id, current_result):
    """Analyze longitudinal trends and detect significant changes."""
    longitudinal_data = load_longitudinal_data()
    user_data = longitudinal_data.get(user_id, [])  # Use .get() with default empty list
    
    if len(user_data) < 1:  # Changed from < 2 to < 1 for first assessment
        return {
            'trend': 'baseline',
            'change_detected': False,
            'trend_description': 'This is your first assessment. Future assessments will track changes over time.',
            'recommendations': ['Continue regular assessments to establish baseline patterns'],
            'assessment_count': 0
        }
    
    # Analyze trends over time
    recent_data = user_data[-5:]  # Last 5 assessments
    predictions = [d.get('prediction', 'Unknown') for d in recent_data]
    confidences = [d.get('confidence', 0.0) for d in recent_data]
    
    # Detect trends
    if len(predictions) >= 3:
        recent_trend = predictions[-3:]
        if all(p == 'Dementia' for p in recent_trend):
            trend = 'declining'
            change_detected = True
            trend_description = 'Consistent decline detected over recent assessments'
        elif all(p == 'Typical' for p in recent_trend):
            trend = 'stable'
            change_detected = False
            trend_description = 'Consistent typical patterns maintained'
        else:
            trend = 'fluctuating'
            change_detected = True
            trend_description = 'Variable patterns detected - monitoring recommended'
    else:
        trend = 'insufficient_data'
        change_detected = False
        trend_description = 'Insufficient data for trend analysis'
    
    # Generate recommendations
    recommendations = []
    if trend == 'declining':
        recommendations = [
            'Consider clinical evaluation for comprehensive assessment',
            'Monitor for additional cognitive symptoms',
            'Schedule follow-up assessment in 2-4 weeks'
        ]
    elif trend == 'fluctuating':
        recommendations = [
            'Continue regular monitoring',
            'Note any environmental factors affecting performance',
            'Consider assessment at consistent times of day'
        ]
    else:
        recommendations = [
            'Continue regular assessments',
            'Maintain current cognitive health practices'
        ]
    
    return {
        'trend': trend,
        'change_detected': change_detected,
        'trend_description': trend_description,
        'recommendations': recommendations,
        'assessment_count': len(user_data)
    }

def generate_personalized_insights(result, emotion_features, explainable_feedback, longitudinal_analysis):
    """Generate personalized insights based on all analysis results."""
    insights = []
    
    # Emotion-based insights
    if emotion_features.get('flat_affect', 0.0) > 0.7:
        insights.append("Monotone speech detected - this could indicate emotional changes or cognitive load")
    
    if emotion_features.get('voice_tremor', 0.0) > 0.6:
        insights.append("Voice tremor detected - this may indicate stress, fatigue, or neurological changes")
    
    if emotion_features.get('pause_frequency', 0.0) > 0.5:
        insights.append("Frequent pauses detected - this may indicate word-finding difficulties or cognitive processing")
    
    # Explainable feedback insights
    if explainable_feedback.get('cognitive_load', {}).get('score', 0.0) > 0.7:
        insights.append("High cognitive load detected - speech patterns suggest increased mental effort")
    
    if explainable_feedback.get('fluency', {}).get('score', 0.0) < 0.4:
        insights.append("Speech fluency concerns - fragmented speech patterns detected")
    
    if explainable_feedback.get('articulation', {}).get('score', 0.0) < 0.4:
        insights.append("Articulation concerns - unclear speech patterns detected")
    
    # Longitudinal insights
    if longitudinal_analysis.get('change_detected', False):
        if longitudinal_analysis.get('trend') == 'declining':
            insights.append("Declining trend detected - consistent changes in speech patterns over time")
        elif longitudinal_analysis.get('trend') == 'fluctuating':
            insights.append("Variable patterns detected - inconsistent speech patterns may indicate cognitive changes")
    
    # Positive insights
    if result.get('prediction') == 'Typical' and result.get('confidence', 0.0) > 0.8:
        insights.append("Strong typical speech patterns detected - cognitive function appears normal")
    
    if emotion_features.get('valence', 0.0) > 0.5:
        insights.append("Positive emotional tone detected - this is associated with better cognitive function")
    
    if explainable_feedback.get('speech_clarity', {}).get('score', 0.0) > 0.8:
        insights.append("Excellent speech clarity - clear articulation patterns detected")
    
    # If no specific insights, provide general guidance
    if not insights:
        insights.append("Speech patterns are within normal range - continue regular monitoring")
    
    return insights

def to_python_type(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_type(v) for v in obj)
    elif isinstance(obj, (np.integer, np.floating, np.bool_, np.number)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # Catch any other numpy scalar types
        try:
            return obj.item()
        except:
            return str(obj)
    elif hasattr(obj, 'tolist'):  # Catch any other numpy array-like objects
        try:
            return obj.tolist()
        except:
            return str(obj)
    elif str(type(obj)).startswith("<class 'numpy"):  # Catch any numpy type
        try:
            if hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return str(obj)
        except:
            return str(obj)
    else:
        return obj

@app.route('/')
def index():
    return render_template('index.html', medical_conditions=MEDICAL_CONDITIONS, supabase_available=SUPABASE_AVAILABLE)

# Authentication routes
@app.route('/auth/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('auth/signup.html', supabase_available=SUPABASE_AVAILABLE)
    
    if not SUPABASE_AVAILABLE:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        result = user_manager.create_user(email, password, full_name)
        
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({'error': f'Error creating account: {str(e)}'}), 500

@app.route('/auth/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'GET':
        return render_template('auth/signin.html', supabase_available=SUPABASE_AVAILABLE)
    
    if not SUPABASE_AVAILABLE:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        result = user_manager.sign_in(email, password)
        
        if result.get('success'):
            # Store user session
            session['user_id'] = result['user_id']
            session['email'] = result['email']
            session['access_token'] = result['access_token']
            return jsonify(result)
        else:
            return jsonify(result), 401
            
    except Exception as e:
        return jsonify({'error': f'Error signing in: {str(e)}'}), 500

@app.route('/auth/signout', methods=['POST'])
def signout():
    if not SUPABASE_AVAILABLE:
        return jsonify({'error': 'Supabase not configured'}), 500
    
    try:
        # Clear session
        session.clear()
        return jsonify({'success': True, 'message': 'Signed out successfully'})
    except Exception as e:
        return jsonify({'error': f'Error signing out: {str(e)}'}), 500

@app.route('/auth/profile')
def profile():
    if not SUPABASE_AVAILABLE:
        return render_template('auth/profile.html', supabase_available=False)
    
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('signin'))
    
    try:
        # Get user profile
        profile_result = user_manager.get_user_profile(user_id)
        if profile_result.get('success'):
            profile = profile_result['profile']
            
            # Get user assessments
            assessments_result = user_manager.get_user_assessments(user_id)
            assessments = assessments_result.get('assessments', []) if assessments_result.get('success') else []
            
            # Get trends
            trends_result = user_manager.get_assessment_trends(user_id)
            trends = trends_result.get('trends', {}) if trends_result.get('success') else {}
            
            return render_template('auth/profile.html', 
                                supabase_available=True,
                                profile=profile,
                                assessments=assessments,
                                trends=trends)
        else:
            return render_template('auth/profile.html', 
                                supabase_available=True,
                                error='Failed to load profile')
    except Exception as e:
        return render_template('auth/profile.html', 
                            supabase_available=True,
                            error=f'Error loading profile: {str(e)}')

@app.route('/auth/check')
def check_auth():
    """Check if user is authenticated."""
    user_id = session.get('user_id')
    if user_id and SUPABASE_AVAILABLE:
        return jsonify({
            'authenticated': True,
            'user_id': user_id,
            'email': session.get('email')
        })
    else:
        return jsonify({'authenticated': False})

@app.route('/record', methods=['POST'])
def record_audio():
    """Handle audio recording from the web interface."""
    print("=== STARTING AUDIO ANALYSIS ===")
    
    # Check if user is authenticated (if Supabase is available)
    user_id = None
    if SUPABASE_AVAILABLE:
        user_id = session.get('user_id')
        if user_id:
            print(f"Authenticated user: {user_id}")
        else:
            print("No authenticated user - using anonymous session")
    
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
        
        # === FEATURE 1: LONGITUDINAL TRACKING ===
        try:
            print("=== ADDING LONGITUDINAL TRACKING ===")
            
            if SUPABASE_AVAILABLE and user_id:
                # Use Supabase for authenticated users
                print(f"Using Supabase for user {user_id}")
                
                # Save assessment to Supabase
                assessment_data = {
                    'prediction': adjusted_result.get('prediction', 'Unknown'),
                    'confidence': adjusted_result.get('confidence', 0.0),
                    'typical_probability': adjusted_result.get('typical_probability', 0.0),
                    'dementia_probability': adjusted_result.get('dementia_probability', 0.0),
                    'medical_history': json.dumps(medical_history),
                    'audio_duration': 5.0
                }
                
                save_result = user_manager.save_assessment(user_id, assessment_data)
                if save_result.get('success'):
                    print(f"✅ Assessment saved to Supabase for user {user_id}")
                else:
                    print(f"⚠️ Failed to save assessment to Supabase: {save_result.get('error')}")
                
                # Get trends from Supabase
                trends_result = user_manager.get_assessment_trends(user_id)
                if trends_result.get('success'):
                    longitudinal_analysis = trends_result['trends']
                    print("✅ Longitudinal analysis from Supabase")
                else:
                    print(f"⚠️ Failed to get trends from Supabase: {trends_result.get('error')}")
                    longitudinal_analysis = {
                        'trend': 'baseline',
                        'change_detected': False,
                        'trend_description': 'Longitudinal tracking unavailable',
                        'recommendations': ['Continue regular assessments'],
                        'assessment_count': 0
                    }
            else:
                # Use local storage for anonymous users
                print("Using local storage for anonymous user")
                
                # Generate or get user ID for longitudinal tracking
                local_user_id = request.form.get('user_id', str(uuid.uuid4()))
                if not local_user_id or local_user_id == 'undefined':
                    local_user_id = str(uuid.uuid4())
                print(f"Using local user_id: {local_user_id}")
                
                # Load existing longitudinal data
                longitudinal_data = load_longitudinal_data()
                
                # Create assessment data
                assessment_data = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction': adjusted_result.get('prediction', 'Unknown'),
                    'confidence': adjusted_result.get('confidence', 0.0),
                    'medical_history': medical_history
                }
                
                # Initialize user data if it doesn't exist
                if local_user_id not in longitudinal_data:
                    print(f"Initializing new user data for {local_user_id}")
                    longitudinal_data[local_user_id] = []
                
                # Add current assessment
                longitudinal_data[local_user_id].append(assessment_data)
                print(f"Saved assessment data for user {local_user_id}")
                
                # Save longitudinal data
                save_longitudinal_data(longitudinal_data)
                
                # Analyze trends
                longitudinal_analysis = analyze_longitudinal_trends(local_user_id, adjusted_result)
                adjusted_result['user_id'] = local_user_id
            
            adjusted_result['longitudinal_analysis'] = longitudinal_analysis
            print("✅ Longitudinal tracking added successfully")
            
        except Exception as e:
            print(f"⚠️ Longitudinal tracking failed: {e}")
            # Don't break the main functionality - just add a default
            adjusted_result['longitudinal_analysis'] = {
                'trend': 'baseline',
                'change_detected': False,
                'trend_description': 'Longitudinal tracking unavailable',
                'recommendations': ['Continue regular assessments'],
                'assessment_count': 0
            }
            if not user_id:
                adjusted_result['user_id'] = str(uuid.uuid4())
        
        # === FEATURE 2: EMOTION AND SENTIMENT ANALYSIS ===
        try:
            print("=== ADDING EMOTION AND SENTIMENT ANALYSIS ===")
            emotion_features = analyze_emotion_and_sentiment(audio_path)
            adjusted_result['emotion_analysis'] = emotion_features
            print("✅ Emotion and sentiment analysis added successfully")
            
        except Exception as e:
            print(f"⚠️ Emotion analysis failed: {e}")
            # Don't break the main functionality - just add a default
            adjusted_result['emotion_analysis'] = {
                'valence': 0.0,
                'arousal': 0.0,
                'flat_affect': 0.0,
                'speech_rate': 0.0,
                'pause_frequency': 0.0,
                'voice_tremor': 0.0,
                'breathing_pattern': 0.0
            }
        
        # === FEATURE 3: EXPLAINABLE AI FEEDBACK ===
        try:
            print("=== ADDING EXPLAINABLE AI FEEDBACK ===")
            explainable_feedback = generate_explainable_feedback(
                audio_path, 
                adjusted_result.get('prediction', 'Unknown'), 
                adjusted_result.get('confidence', 0.0)
            )
            adjusted_result['explainable_feedback'] = explainable_feedback
            print("✅ Explainable AI feedback added successfully")
            
        except Exception as e:
            print(f"⚠️ Explainable feedback failed: {e}")
            # Don't break the main functionality - just add a default
            adjusted_result['explainable_feedback'] = {
                'speech_clarity': {'score': 0.5, 'feedback': 'Analysis unavailable'},
                'fluency': {'score': 0.5, 'feedback': 'Analysis unavailable'},
                'prosody': {'score': 0.5, 'feedback': 'Analysis unavailable'},
                'articulation': {'score': 0.5, 'feedback': 'Analysis unavailable'},
                'voice_quality': {'score': 0.5, 'feedback': 'Analysis unavailable'},
                'cognitive_load': {'score': 0.5, 'feedback': 'Analysis unavailable'}
            }
        
        # === FEATURE 4: PERSONALIZED INSIGHTS ===
        try:
            print("=== ADDING PERSONALIZED INSIGHTS ===")
            insights = generate_personalized_insights(
                adjusted_result, 
                adjusted_result.get('emotion_analysis', {}), 
                adjusted_result.get('explainable_feedback', {}), 
                adjusted_result.get('longitudinal_analysis', {})
            )
            adjusted_result['personalized_insights'] = insights
            print("✅ Personalized insights added successfully")
            
        except Exception as e:
            print(f"⚠️ Personalized insights failed: {e}")
            # Don't break the main functionality - just add a default
            adjusted_result['personalized_insights'] = ["Analysis completed successfully"]
        
        print("=== ANALYSIS COMPLETE ===")
        
        # Debug: Print what's being returned
        print("=== DEBUG: FINAL RESPONSE STRUCTURE ===")
        print(f"Keys in adjusted_result: {list(adjusted_result.keys())}")
        if 'longitudinal_analysis' in adjusted_result:
            print(f"Longitudinal analysis: {adjusted_result['longitudinal_analysis']}")
        if 'emotion_analysis' in adjusted_result:
            print(f"Emotion analysis keys: {list(adjusted_result['emotion_analysis'].keys())}")
        if 'explainable_feedback' in adjusted_result:
            print(f"Explainable feedback keys: {list(adjusted_result['explainable_feedback'].keys())}")
        if 'personalized_insights' in adjusted_result:
            print(f"Personalized insights: {adjusted_result['personalized_insights']}")
        print("=== END DEBUG ===")
        
        return jsonify(to_python_type(adjusted_result))
        
    except Exception as e:
        print(f"Error in record_audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500
        
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
        # Run inference script using the same approach as record_and_detect.py
        import sys
        result = subprocess.run(
            [sys.executable, 'inference.py', audio_path],
            capture_output=True,
            text=True
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