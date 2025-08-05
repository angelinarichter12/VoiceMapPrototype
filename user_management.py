import os
from datetime import datetime, timedelta
from supabase_config import get_supabase_client, is_supabase_available
import json

class UserManager:
    def __init__(self):
        self.supabase = get_supabase_client()
        self.is_available = is_supabase_available()
    
    def create_user(self, email, password, full_name=None):
        """Create a new user account."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            # Create user in Supabase Auth
            auth_response = self.supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                user_id = auth_response.user.id
                
                # Create user profile in profiles table
                profile_data = {
                    'id': user_id,
                    'email': email,
                    'full_name': full_name or '',
                    'created_at': datetime.now().isoformat(),
                    'last_assessment': None,
                    'assessment_count': 0
                }
                
                self.supabase.table('profiles').insert(profile_data).execute()
                
                return {
                    'success': True,
                    'user_id': user_id,
                    'email': email,
                    'message': 'User created successfully'
                }
            else:
                return {'error': 'Failed to create user'}
                
        except Exception as e:
            return {'error': f'Error creating user: {str(e)}'}
    
    def sign_in(self, email, password):
        """Sign in a user."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            auth_response = self.supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                return {
                    'success': True,
                    'user_id': auth_response.user.id,
                    'email': auth_response.user.email,
                    'access_token': auth_response.session.access_token,
                    'refresh_token': auth_response.session.refresh_token
                }
            else:
                return {'error': 'Invalid credentials'}
                
        except Exception as e:
            return {'error': f'Error signing in: {str(e)}'}
    
    def sign_out(self, access_token):
        """Sign out a user."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            self.supabase.auth.sign_out()
            return {'success': True, 'message': 'Signed out successfully'}
        except Exception as e:
            return {'error': f'Error signing out: {str(e)}'}
    
    def get_user_profile(self, user_id):
        """Get user profile information."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            response = self.supabase.table('profiles').select('*').eq('id', user_id).execute()
            
            if response.data:
                return {'success': True, 'profile': response.data[0]}
            else:
                return {'error': 'User profile not found'}
                
        except Exception as e:
            return {'error': f'Error getting user profile: {str(e)}'}
    
    def update_user_profile(self, user_id, profile_data):
        """Update user profile information."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            response = self.supabase.table('profiles').update(profile_data).eq('id', user_id).execute()
            
            if response.data:
                return {'success': True, 'profile': response.data[0]}
            else:
                return {'error': 'Failed to update profile'}
                
        except Exception as e:
            return {'error': f'Error updating profile: {str(e)}'}
    
    def save_assessment(self, user_id, assessment_data):
        """Save a new assessment for a user."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            # Add user_id and timestamp to assessment data
            assessment_data['user_id'] = user_id
            assessment_data['created_at'] = datetime.now().isoformat()
            
            # Insert assessment
            response = self.supabase.table('assessments').insert(assessment_data).execute()
            
            if response.data:
                # Update user's last_assessment and assessment_count
                user_response = self.supabase.table('profiles').select('assessment_count').eq('id', user_id).execute()
                
                if user_response.data:
                    current_count = user_response.data[0].get('assessment_count', 0)
                    new_count = current_count + 1
                    
                    self.supabase.table('profiles').update({
                        'last_assessment': datetime.now().isoformat(),
                        'assessment_count': new_count
                    }).eq('id', user_id).execute()
                
                return {'success': True, 'assessment_id': response.data[0]['id']}
            else:
                return {'error': 'Failed to save assessment'}
                
        except Exception as e:
            return {'error': f'Error saving assessment: {str(e)}'}
    
    def get_user_assessments(self, user_id, limit=10):
        """Get user's assessment history."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            response = self.supabase.table('assessments').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
            
            return {'success': True, 'assessments': response.data}
                
        except Exception as e:
            return {'error': f'Error getting assessments: {str(e)}'}
    
    def get_assessment_trends(self, user_id):
        """Get assessment trends for a user."""
        if not self.is_available:
            return {'error': 'Supabase not configured'}
        
        try:
            # Get all assessments for the user
            response = self.supabase.table('assessments').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
            
            if not response.data:
                return {
                    'success': True,
                    'trends': {
                        'assessment_count': 0,
                        'trend': 'baseline',
                        'change_detected': False,
                        'trend_description': 'No assessments yet',
                        'recommendations': ['Start your first assessment']
                    }
                }
            
            assessments = response.data
            
            # Analyze trends
            predictions = [a.get('prediction', 'Unknown') for a in assessments]
            confidences = [a.get('confidence', 0.0) for a in assessments]
            
            # Calculate trend
            if len(assessments) >= 3:
                recent_confidences = confidences[:3]
                if all(c > 0.7 for c in recent_confidences):
                    trend = 'improving'
                    trend_description = 'Recent assessments show consistent results'
                elif any(c < 0.5 for c in recent_confidences):
                    trend = 'declining'
                    trend_description = 'Recent assessments show some concerns'
                else:
                    trend = 'stable'
                    trend_description = 'Assessments show stable patterns'
            else:
                trend = 'baseline'
                trend_description = 'Establishing baseline patterns'
            
            return {
                'success': True,
                'trends': {
                    'assessment_count': len(assessments),
                    'trend': trend,
                    'change_detected': len(assessments) > 1,
                    'trend_description': trend_description,
                    'recommendations': self._generate_recommendations(trend, len(assessments))
                }
            }
                
        except Exception as e:
            return {'error': f'Error getting trends: {str(e)}'}
    
    def _generate_recommendations(self, trend, assessment_count):
        """Generate personalized recommendations based on trends."""
        recommendations = []
        
        if assessment_count == 0:
            recommendations.append('Start your first assessment to establish baseline')
        elif assessment_count < 3:
            recommendations.append('Continue regular assessments to establish patterns')
        else:
            if trend == 'improving':
                recommendations.append('Excellent progress! Continue regular assessments')
                recommendations.append('Consider sharing results with healthcare provider')
            elif trend == 'stable':
                recommendations.append('Maintain regular assessment schedule')
                recommendations.append('Monitor for any changes in patterns')
            elif trend == 'declining':
                recommendations.append('Consider consulting with healthcare provider')
                recommendations.append('Increase assessment frequency')
                recommendations.append('Review lifestyle factors that may affect cognitive health')
        
        recommendations.append('Maintain healthy lifestyle habits')
        return recommendations

# Global user manager instance
user_manager = UserManager() 