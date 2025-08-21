import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
import os
from django.conf import settings

class FraudDetectionService:
    def __init__(self):
        # Path to your trained models (you'll create these later)
        self.model_dir = os.path.join(settings.BASE_DIR, 'models')
        
        # Try to load trained models if they exist
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                if os.path.exists(features_path):
                    self.feature_names = joblib.load(features_path)
                else:
                    self.feature_names = self._get_default_features()
                self.is_trained = True
                print("✅ Loaded pre-trained fraud detection model")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
                self.vectorizer = None
                self.feature_names = self._get_default_features()
                self.is_trained = False
        else:
            print("⚠️ No pre-trained model found. Using rule-based detection.")
            self.model = None
            self.vectorizer = None
            self.feature_names = self._get_default_features()
            self.is_trained = False
    
    def _get_default_features(self):
        return [
            'desc_sentiment', 'desc_subjectivity', 'req_sentiment', 'req_subjectivity',
            'company_sentiment', 'company_subjectivity', 'title_length', 'description_length',
            'requirements_length', 'title_word_count', 'description_word_count',
            'missing_salary', 'missing_company', 'missing_requirements',
            'has_company_logo', 'has_questions', 'telecommuting'
        ]
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        # Remove URLs and special patterns
        text = re.sub(r'#URL_[a-f0-9]+#', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    def get_sentiment(self, text):
        """Extract sentiment polarity and subjectivity from text"""
        if not text or pd.isna(text):
            return 0.0, 0.0
        
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0, 0.0
    
    def extract_features(self, job_data):
        """Extract all features from job posting data"""
        # Text sentiment analysis
        desc_sent, desc_subj = self.get_sentiment(job_data.get('description', ''))
        req_sent, req_subj = self.get_sentiment(job_data.get('requirements', ''))
        comp_sent, comp_subj = self.get_sentiment(job_data.get('company_profile', ''))
        
        # Text length features
        title = job_data.get('title', '')
        description = job_data.get('description', '')
        requirements = job_data.get('requirements', '')
        
        # Feature extraction
        features = {
            # Sentiment features
            'desc_sentiment': desc_sent,
            'desc_subjectivity': desc_subj,
            'req_sentiment': req_sent,
            'req_subjectivity': req_subj,
            'company_sentiment': comp_sent,
            'company_subjectivity': comp_subj,
            
            # Text length features
            'title_length': len(title) if title else 0,
            'description_length': len(description) if description else 0,
            'requirements_length': len(requirements) if requirements else 0,
            'title_word_count': len(title.split()) if title else 0,
            'description_word_count': len(description.split()) if description else 0,
            
            # Missing information flags
            'missing_salary': 1 if not job_data.get('salary_range') else 0,
            'missing_company': 1 if not job_data.get('company_profile') else 0,
            'missing_requirements': 1 if not job_data.get('requirements') else 0,
            
            # Binary features
            'has_company_logo': 1 if job_data.get('has_company_logo') else 0,
            'has_questions': 1 if job_data.get('has_questions') else 0,
            'telecommuting': 1 if job_data.get('telecommuting') else 0,
        }
        
        return features
    
    def rule_based_prediction(self, features):
        """Simple rule-based fraud detection as fallback"""
        fraud_score = 0.0
        confidence = 0.7
        
        # Rule 1: Missing critical information
        if features['missing_company']:
            fraud_score += 0.35
        if features['missing_salary']:
            fraud_score += 0.25
        if features['missing_requirements']:
            fraud_score += 0.15
        
        # Rule 2: Suspicious text characteristics
        if features['description_length'] < 50:
            fraud_score += 0.30
        elif features['description_length'] < 100:
            fraud_score += 0.15
        
        if features['title_length'] < 5:
            fraud_score += 0.20
        
        # Rule 3: Sentiment analysis
        if features['desc_sentiment'] < -0.3:
            fraud_score += 0.25
        elif features['desc_sentiment'] < -0.1:
            fraud_score += 0.10
        
        # Rule 4: Company credibility indicators
        if not features['has_company_logo']:
            fraud_score += 0.15
        if not features['has_questions']:
            fraud_score += 0.10
        
        # Rule 5: Overly positive or generic descriptions
        if features['desc_sentiment'] > 0.8:
            fraud_score += 0.20
        if features['desc_subjectivity'] > 0.8:
            fraud_score += 0.15
        
        # Normalize score between 0 and 1
        fraud_probability = min(fraud_score, 1.0)
        
        return fraud_probability, confidence
    
    def ml_prediction(self, job_data):
        """Machine learning based prediction (when model is available)"""
        try:
            # Extract features
            features = self.extract_features(job_data)
            
            # Prepare text data
            combined_text = ' '.join([
                job_data.get('title', ''),
                job_data.get('description', ''),
                job_data.get('requirements', ''),
                job_data.get('company_profile', '')
            ])
            cleaned_text = self.clean_text(combined_text)
            
            # Vectorize text
            text_features = self.vectorizer.transform([cleaned_text])
            
            # Combine numerical and text features
            numerical_values = [features[name] for name in self.feature_names]
            X = np.hstack([text_features.toarray(), [numerical_values]])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)
            
            fraud_probability = probabilities[1] if len(probabilities) > 1 else probabilities
            confidence = probabilities.max()
            
            return fraud_probability, confidence
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fallback to rule-based prediction
            return self.rule_based_prediction(self.extract_features(job_data))
    
    def predict_fraud(self, job_data):
        """Main prediction function"""
        try:
            # Extract features for analysis
            features = self.extract_features(job_data)
            
            # Use ML model if available, otherwise use rule-based approach
            if self.is_trained and self.model is not None:
                fraud_probability, confidence = self.ml_prediction(job_data)
                method = "ML Model"
            else:
                fraud_probability, confidence = self.rule_based_prediction(features)
                method = "Rule-based"
            
            # Determine if fraudulent
            is_fraudulent = fraud_probability > 0.5
            
            # Determine risk level
            if fraud_probability >= 0.7:
                risk_level = 'High'
            elif fraud_probability >= 0.4:
                risk_level = 'Medium'
            else:
                risk_level = 'Low'
            
            # Return comprehensive result
            result = {
                'is_fraudulent': is_fraudulent,
                'confidence': confidence,
                'fraud_probability': fraud_probability,
                'sentiment_score': features['desc_sentiment'],
                'risk_level': risk_level,
                'method': method,
                'features': features  # Include features for debugging
            }
            
            print(f"🔍 Fraud prediction completed using {method}")
            print(f"   Fraud Probability: {fraud_probability:.2%}")
            print(f"   Risk Level: {risk_level}")
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Return safe default
            return {
                'is_fraudulent': False,
                'confidence': 0.5,
                'fraud_probability': 0.3,
                'sentiment_score': 0.0,
                'risk_level': 'Low',
                'method': 'Error fallback',
                'error': str(e)
            }
