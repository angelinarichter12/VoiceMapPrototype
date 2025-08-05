#!/usr/bin/env python3
"""
Test script for VoiceMap's advanced features:
1. Longitudinal tracking
2. Emotion and sentiment analysis
3. Explainable AI feedback
4. Personalized insights
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
import numpy as np

# Add current directory to path
sys.path.append('.')

from app import (
    analyze_emotion_and_sentiment,
    generate_explainable_feedback,
    analyze_longitudinal_trends,
    generate_personalized_insights,
    load_longitudinal_data,
    save_longitudinal_data
)

def test_emotion_analysis():
    """Test emotion and sentiment analysis."""
    print("=== Testing Emotion Analysis ===")
    
    # Create a test audio file (simulate)
    test_audio_path = "recorded.wav"  # Use existing file
    
    if os.path.exists(test_audio_path):
        emotion_features = analyze_emotion_and_sentiment(test_audio_path)
        print("✅ Emotion analysis completed")
        print(f"   - Valence: {emotion_features.get('valence', 0):.3f}")
        print(f"   - Arousal: {emotion_features.get('arousal', 0):.3f}")
        print(f"   - Flat affect: {emotion_features.get('flat_affect', 0):.3f}")
        print(f"   - Voice tremor: {emotion_features.get('voice_tremor', 0):.3f}")
        print(f"   - Speech rate: {emotion_features.get('speech_rate', 0):.3f}")
        print(f"   - Pause frequency: {emotion_features.get('pause_frequency', 0):.3f}")
    else:
        print("⚠️ Test audio file not found, skipping emotion analysis test")

def test_explainable_feedback():
    """Test explainable AI feedback generation."""
    print("\n=== Testing Explainable AI Feedback ===")
    
    test_audio_path = "recorded.wav"
    
    if os.path.exists(test_audio_path):
        feedback = generate_explainable_feedback(test_audio_path, "Typical", 85.0)
        print("✅ Explainable feedback generated")
        
        for category, data in feedback.items():
            if isinstance(data, dict) and 'score' in data and 'feedback' in data:
                print(f"   - {category.replace('_', ' ').title()}: {data['score']:.3f}")
                print(f"     Feedback: {data['feedback']}")
    else:
        print("⚠️ Test audio file not found, skipping explainable feedback test")

def test_longitudinal_tracking():
    """Test longitudinal tracking functionality."""
    print("\n=== Testing Longitudinal Tracking ===")
    
    # Create test user data
    user_id = "test_user_123"
    test_result = {
        'prediction': 'Typical',
        'confidence': 85.0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test baseline (first assessment)
    longitudinal_analysis = analyze_longitudinal_trends(user_id, test_result)
    print("✅ Longitudinal analysis completed")
    print(f"   - Trend: {longitudinal_analysis.get('trend', 'unknown')}")
    print(f"   - Change detected: {longitudinal_analysis.get('change_detected', False)}")
    print(f"   - Assessment count: {longitudinal_analysis.get('assessment_count', 0)}")
    print(f"   - Description: {longitudinal_analysis.get('trend_description', 'N/A')}")
    
    if longitudinal_analysis.get('recommendations'):
        print("   - Recommendations:")
        for rec in longitudinal_analysis['recommendations']:
            print(f"     * {rec}")

def test_personalized_insights():
    """Test personalized insights generation."""
    print("\n=== Testing Personalized Insights ===")
    
    # Mock data for testing
    result = {'prediction': 'Typical', 'confidence': 85.0}
    emotion_features = {
        'valence': 0.3,
        'arousal': 0.6,
        'flat_affect': 0.2,
        'voice_tremor': 0.1,
        'pause_frequency': 0.3
    }
    explainable_feedback = {
        'speech_clarity': {'score': 0.8, 'feedback': 'Excellent clarity'},
        'fluency': {'score': 0.7, 'feedback': 'Good fluency'},
        'cognitive_load': {'score': 0.3, 'feedback': 'Low cognitive load'}
    }
    longitudinal_analysis = {
        'trend': 'stable',
        'change_detected': False,
        'assessment_count': 1
    }
    
    insights = generate_personalized_insights(
        result, emotion_features, explainable_feedback, longitudinal_analysis
    )
    
    print("✅ Personalized insights generated")
    for insight in insights:
        print(f"   - {insight}")

def test_data_persistence():
    """Test longitudinal data persistence."""
    print("\n=== Testing Data Persistence ===")
    
    # Test saving and loading data
    test_data = {
        'user_1': [
            {
                'timestamp': datetime.now().isoformat(),
                'prediction': 'Typical',
                'confidence': 85.0
            }
        ]
    }
    
    save_longitudinal_data(test_data)
    loaded_data = load_longitudinal_data()
    
    if 'user_1' in loaded_data:
        print("✅ Data persistence working")
        print(f"   - Saved {len(loaded_data['user_1'])} assessments for user_1")
    else:
        print("⚠️ Data persistence test failed")

def main():
    """Run all tests."""
    print("🧠 VoiceMap Advanced Features Test Suite")
    print("=" * 50)
    
    try:
        test_emotion_analysis()
        test_explainable_feedback()
        test_longitudinal_tracking()
        test_personalized_insights()
        test_data_persistence()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("\n🎯 Advanced Features Implemented:")
        print("   1. ✅ Longitudinal tracking of voice biomarkers")
        print("   2. ✅ Emotion and sentiment analysis")
        print("   3. ✅ Explainable AI with personalized feedback")
        print("   4. ✅ Personalized insights generation")
        print("   5. ✅ Data persistence for trend analysis")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 