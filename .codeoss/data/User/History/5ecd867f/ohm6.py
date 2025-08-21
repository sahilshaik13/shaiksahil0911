import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
import os
from django.conf import settings

class FraudDetectionService:
    def __init__(self):
        # For now, we'll create a simple mock service
        # Later you can replace this with your trained model
        self.model = None
        self.vectorizer = None
        self.feature_names = [
            'desc_sentiment', 'desc_subjectivity', 'req_sentiment', 'req_subjectivity',
            'company_sentiment', 'company_subjectivity', 'title_length', 'description_length',
            'requirements_length', 'title_word_count', 'description_word_count',
            'missing_salary', 'missing_company', 'missing_requirements',
            'has_company_logo', 'has_questions', 'telecommuting'
        ]
    
    def clean_text(self, text):
        if pd.isna(text) or not text:
            return ""
        text = re.sub(r'#URL_[a-f0-9]+#', '', str(text))
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()
    
    def get_sentiment(self, text):
        if not text:
            return 0, 0
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except:
            return 0, 0
    
    def predict_fraud(self, job_data):
        # Extract sentiment features
        desc_sent, desc_subj = self.get_sentiment(job_data.get('description', ''))
        req_sent, req_subj = self.get_sentiment(job_data.get('requirements', ''))
        comp_sent, comp_subj = self.get_sentiment(job_data.get('company_profile', ''))
        
        # Create feature vector
        features = {
            'desc_sentiment': desc_sent,
            'desc_subjectivity': desc_subj,
            'req_sentiment': req_sent,
            'req_subjectivity': req_subj,
            'company_sentiment': comp_sent,
            'company_subjectivity': comp_subj,
            'title_length': len(job_data.get('title', '')),
            'description_length': len(job_data.get('description', '')),
            'requirements_length': len(job_data.get('requirements', '')),
            'title_word_count': len(job_data.get('title', '').split()),
            'description_word_count': len(job_data.get('description', '').split()),
            'missing_salary': 1 if not job_data.get('salary_range') else 0,
            'missing_company': 1 if not job_data.get('company_profile') else 0,
            'missing_requirements': 1 if not job_data.get('requirements') else 0,
            'has_company_logo': 1 if job_data.get('has_company_logo') else 0,
            'has_questions': 1 if job_data.get('has_questions') else 0,
            'telecommuting': 1 if job_data.get('telecommuting') else 0,
        }
        
        # MOCK PREDICTION LOGIC (Replace with your trained model later)
        # This is a simple rule-based system for demonstration
        fraud_score = 0
        
        # Check for suspicious patterns
        if features['missing_company']:
            fraud_score += 0.3
        if features['missing_salary']:
            fraud_score += 0.2
        if features['desc_sentiment'] < -0.2:
            fraud_score += 0.2
        if features['title_length'] < 10:
            fraud_score += 0.1
        if features['description_length'] < 100:
            fraud_score += 0.2
        if not features['has_company_logo']:
            fraud_score += 0.1
        
        # Normalize score
        fraud_probability = min(fraud_score, 1.0)
        confidence = 0.8  # Mock confidence
        
        # Determine if fraudulent
        is_fraudulent = fraud_probability > 0.5
        
        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = 'High'
        elif fraud_probability >= 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'is_fraudulent': is_fraudulent,
            'confidence': confidence,
            'fraud_probability': fraud_probability,
            'sentiment_score': desc_sent,
            'risk_level': risk_level
        }

# You can add your trained model loading here later:
# class FraudDetectionService:
#     def __init__(self):
#         model_path = os.path.join(settings.BASE_DIR, 'models', 'fraud_detection_model.pkl')
#         vectorizer_path = os.path.join(settings.BASE_DIR, 'models', 'tfidf_vectorizer.pkl')
#         
#         if os.path.exists(model_path):
#             self.model = joblib.load(model_path)
#             self.vectorizer = joblib.load(vectorizer_path)
#         else:
#             self.model = None
#             self.vectorizer = None
