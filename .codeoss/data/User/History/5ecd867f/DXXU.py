import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
import os
from django.conf import settings
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from sklearn.metrics.pairwise import cosine_similarity


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
        
        # Set up Vertex AI (assumes GOOGLE_APPLICATION_CREDENTIALS env var is set)
        try:
            aiplatform.init(location='asia-south1')  # Change region as needed
            self.endpoint = aiplatform.Endpoint('projects/your-project-id/locations/asia-south1/endpoints/your-endpoint-id')  # Replace with your Vertex AI endpoint
            # Predefined fraud patterns (embeddings; in practice, generate these from known fraud data)
            self.fraud_patterns = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # Dummy; replace with real embeddings (e.g., 768-dim)
            self.vertex_available = True
            print("✅ Vertex AI initialized successfully")
        except Exception as e:
            print(f"❌ Vertex AI initialization error: {e}. Falling back to local methods.")
            self.vertex_available = False
            self.endpoint = None
            self.fraud_patterns = None
    
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
        """Extract sentiment polarity and subjectivity from text (fallback to TextBlob if Vertex AI unavailable)"""
        if not text or pd.isna(text):
            return 0.0, 0.0
        
        if self.vertex_available:
            try:
                response = self._vertex_predict(text, task='sentiment')
                sentiment_score = response.get('sentiment', 0.0)  # Assuming -1 to 1 polarity
                subjectivity = response.get('subjectivity', 0.0)  # Assuming 0-1 subjectivity
                return sentiment_score, subjectivity
            except Exception as e:
                print(f"Vertex AI sentiment error: {e}. Falling back to TextBlob.")
        
        # Fallback to TextBlob
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
            probabilities = self.model.predict_proba(X)[0]  # Assuming binary classification
            
            fraud_probability = probabilities[1]  # Probability of fraud class
            confidence = max(probabilities)  # Max probability as confidence
            
            return fraud_probability, confidence
            
        except Exception as e:
            print(f"ML prediction error: {e}")
            # Fallback to rule-based prediction
            return self.rule_based_prediction(self.extract_features(job_data))
    
    def _vertex_predict(self, text, task):
        """Helper to call Vertex AI (simplified; customize prompts)"""
        if task == 'sentiment':
            prompt = f"Analyze sentiment of this job posting (return polarity (-1 to 1) and subjectivity (0 to 1)): {text}"
        elif task == 'classify_ai_generation':
            prompt = f"Is this text likely AI-generated? Respond with probability (0-1): {text}"
        elif task == 'embedding':
            prompt = f"Generate embedding vector for: {text}"
        
        instances = [json_format.ParseDict({"content": prompt, "parameters": {"maxOutputTokens": 256}}, Value())]
        response = self.endpoint.predict(instances=instances).predictions[0]
        return json_format.MessageToDict(response)  # Parse response to dict
    
    def predict_fraud(self, job_data):
        """Main prediction function with Vertex AI integration and complex calculations"""
        try:
            # Extract features for analysis
            features = self.extract_features(job_data)
            
            # Prepare combined text for Vertex AI
            combined_text = ' '.join([
                job_data.get('title', ''),
                job_data.get('description', ''),
                job_data.get('requirements', ''),
                job_data.get('company_profile', '')
            ])
            cleaned_text = self.clean_text(combined_text)
            
            # Step 1: Get base prediction (ML or rule-based)
            if self.is_trained and self.model is not None:
                base_prob, base_conf = self.ml_prediction(job_data)
                method = "ML Model"
            else:
                base_prob, base_conf = self.rule_based_prediction(features)
                method = "Rule-based"
            
            # Step 2: Vertex AI enhancements (if available)
            sentiment_score = features['desc_sentiment']  # Default from local
            ai_generation_score = 0.0
            embedding_similarity = 0.0
            
            if self.vertex_available:
                try:
                    # Sentiment (override local if Vertex succeeds)
                    sentiment_response = self._vertex_predict(cleaned_text, task='sentiment')
                    sentiment_score = sentiment_response.get('sentiment', sentiment_score)  # -1 to 1
                    
                    # AI-generated content detection
                    ai_gen_response = self._vertex_predict(cleaned_text, task='classify_ai_generation')
                    ai_generation_score = ai_gen_response.get('ai_prob', 0.0)  # 0-1
                    
                    # Embeddings similarity to fraud patterns
                    embedding_response = self._vertex_predict(cleaned_text, task='embedding')
                    embedding = np.array(embedding_response.get('embedding', [0.0] * 768))  # Assuming 768-dim
                    similarities = cosine_similarity([embedding], self.fraud_patterns)
                    embedding_similarity = np.max(similarities)  # Highest similarity to known fraud
                except Exception as e:
                    print(f"Vertex AI error: {e}. Using local fallbacks.")
            
            # Step 3: Complex calculation for fraud probability (weighted average)
            fraud_probability = (
                0.4 * base_prob +  # Base model weight
                0.3 * embedding_similarity +  # Embeddings weight
                0.2 * (1 - sentiment_score if sentiment_score > 0 else abs(sentiment_score)) +  # Penalize extremes
                0.1 * ai_generation_score  # AI generation weight
            )
            fraud_probability = min(max(fraud_probability, 0.0), 1.0)  # Clamp to 0-1
            
            # Step 4: Confidence score (adjusted by variance)
            scores = [base_prob, embedding_similarity, abs(sentiment_score), ai_generation_score]
            variance = np.var(scores)
            confidence = base_conf * (1 - variance)  # Reduce if high variance
            confidence = min(max(confidence, 0.0), 1.0)  # Clamp to 0-1
            
            # Step 5: Determine if fraudulent and risk level
            is_fraudulent = fraud_probability > 0.5
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
                'sentiment_score': sentiment_score,
                'risk_level': risk_level,
                'method': f"{method} + Vertex AI" if self.vertex_available else method,
                'features': features,  # Include features for debugging
                'ai_generation_score': ai_generation_score,
                'embedding_similarity': embedding_similarity
            }
            
            print(f"🔍 Fraud prediction completed using {result['method']}")
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
                'features': {},
                'ai_generation_score': 0.0,
                'embedding_similarity': 0.0,
                'error': str(e)
            }
