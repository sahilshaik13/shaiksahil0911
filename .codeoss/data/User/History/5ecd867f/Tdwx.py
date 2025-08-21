import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
import os
import time
import json
import base64
import io
import requests
from PIL import Image
from django.conf import settings

# Google Cloud imports
try:
    from google.cloud import aiplatform
    from google.cloud import vision
    from google.cloud import language_v1
    from google.oauth2 import service_account
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    print("⚠️ Google Cloud libraries not installed. Some features will be disabled.")
    GOOGLE_CLOUD_AVAILABLE = False


class EnhancedFraudDetectionService:
    """Enhanced fraud detection service with Vertex AI integration"""
    
    def __init__(self):
        # Initialize traditional ML components
        self._initialize_traditional_ml()
        
        # Initialize Google Cloud services
        self._initialize_google_cloud()
        
        # Configuration
        self.config = {
            'ai_detection_threshold': 0.7,
            'logo_quality_threshold': 0.6,
            'company_verification_threshold': 0.5,
            'fraud_probability_threshold': 0.6,
            'enable_detailed_logging': getattr(settings, 'FRAUD_DETECTION_DEBUG', False)
        }
        
        # Risk weights for multi-modal prediction
        self.risk_weights = {
            'text': 0.25,
            'logo': 0.20,
            'ai_content': 0.25,
            'company': 0.20,
            'sentiment': 0.10
        }
    
    def _initialize_traditional_ml(self):
        """Initialize traditional ML models"""
        self.model_dir = os.path.join(settings.BASE_DIR, 'models')
        
        # Try to load trained models
        model_path = os.path.join(self.model_dir, 'fraud_detection_model.pkl')
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
        features_path = os.path.join(self.model_dir, 'feature_names.pkl')
        
        if all(os.path.exists(p) for p in [model_path, vectorizer_path]):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                self.feature_names = joblib.load(features_path) if os.path.exists(features_path) else self._get_default_features()
                self.is_trained = True
                print("✅ Loaded pre-trained fraud detection model")
            except Exception as e:
                print(f"❌ Error loading traditional ML model: {e}")
                self._initialize_fallback_ml()
        else:
            print("⚠️ No pre-trained traditional ML model found. Using enhanced rule-based detection.")
            self._initialize_fallback_ml()
    
    def _initialize_fallback_ml(self):
        """Initialize fallback ML components"""
        self.model = None
        self.vectorizer = None
        self.feature_names = self._get_default_features()
        self.is_trained = False
    
    def _initialize_google_cloud(self):
        """Initialize Google Cloud AI services"""
        if not GOOGLE_CLOUD_AVAILABLE:
            self.vertex_ai_enabled = False
            self.vision_client = None
            self.language_client = None
            return
        
        try:
            # Initialize Google Cloud credentials
            project_id = getattr(settings, 'GOOGLE_CLOUD_PROJECT', None)
            region = getattr(settings, 'GOOGLE_CLOUD_REGION', 'us-central1')
            credentials_path = getattr(settings, 'GOOGLE_APPLICATION_CREDENTIALS', None)
            
            if project_id and credentials_path and os.path.exists(credentials_path):
                # Load service account credentials
                credentials = service_account.Credentials.from_service_account_file(credentials_path)
                
                # Initialize clients with credentials
                self.vision_client = vision.ImageAnnotatorClient(credentials=credentials)
                self.language_client = language_v1.LanguageServiceClient(credentials=credentials)
                
                # Initialize Vertex AI
                aiplatform.init(project=project_id, location=region, credentials=credentials)
                
                self.vertex_ai_enabled = True
                print("✅ Google Cloud AI services initialized successfully")
                
            else:
                raise Exception("Missing Google Cloud configuration")
                
        except Exception as e:
            print(f"⚠️ Google Cloud AI services not available: {e}")
            self.vertex_ai_enabled = False
            self.vision_client = None
            self.language_client = None
    
    def _get_default_features(self):
        """Get default feature list"""
        return [
            'desc_sentiment', 'desc_subjectivity', 'req_sentiment', 'req_subjectivity',
            'company_sentiment', 'company_subjectivity', 'title_length', 'description_length',
            'requirements_length', 'title_word_count', 'description_word_count',
            'missing_salary', 'missing_company', 'missing_requirements',
            'has_company_logo', 'has_questions', 'telecommuting'
        ]
    
    # ========== TEXT PROCESSING METHODS ==========
    
    def clean_text(self, text):
        """Enhanced text cleaning and preprocessing"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        
        # Remove URLs and email patterns
        text = re.sub(r'#URL_[a-f0-9]+#', '', text)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        text = re.sub(r'\(\d{3}\)\s*\d{3}[-.]?\d{4}', '', text)
        
        # Clean special characters but preserve meaningful punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.lower().strip()
    
    def get_enhanced_sentiment(self, text):
        """Enhanced sentiment analysis with additional metrics"""
        if not text or pd.isna(text):
            return {'polarity': 0.0, 'subjectivity': 0.0, 'word_count': 0, 'avg_word_length': 0.0}
        
        try:
            blob = TextBlob(str(text))
            words = text.split()
            
            result = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'word_count': len(words),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0.0
            }
            
            return result
            
        except Exception as e:
            if self.config['enable_detailed_logging']:
                print(f"Enhanced sentiment analysis error: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'word_count': 0, 'avg_word_length': 0.0}
    
    # ========== LOGO ANALYSIS METHODS ==========
    
    def analyze_company_logo(self, logo_file):
        """Comprehensive logo analysis using Vision AI"""
        if not self.vertex_ai_enabled or not self.vision_client:
            return self._fallback_logo_analysis(logo_file)
        
        start_time = time.time()
        
        try:
            # Reset file pointer and read content
            if hasattr(logo_file, 'seek'):
                logo_file.seek(0)
            image_content = logo_file.read()
            
            # Create Vision API image object
            image = vision.Image(content=image_content)
            
            logo_analysis = {
                'has_logo': False,
                'professional_quality': 0.0,
                'text_in_logo': [],
                'brand_recognition': None,
                'image_quality_score': 0.0,
                'colors_analysis': {},
                'safe_search': {},
                'object_detection': [],
                'processing_time': 0.0,
                'analysis_confidence': 0.0
            }
            
            # 1. Text detection in logo
            text_response = self.vision_client.text_detection(image=image)
            if text_response.text_annotations:
                logo_analysis['has_logo'] = True
                logo_analysis['text_in_logo'] = [
                    {
                        'text': annotation.description,
                        'confidence': getattr(annotation, 'confidence', 0.9)
                    }
                    for annotation in text_response.text_annotations[:5]
                ]
            
            # 2. Image properties analysis
            properties_response = self.vision_client.image_properties(image=image)
            if properties_response.dominant_colors_annotation.colors:
                logo_analysis['professional_quality'] = self._calculate_design_quality(properties_response)
                logo_analysis['colors_analysis'] = self._analyze_color_scheme(
                    properties_response.dominant_colors_annotation.colors
                )
            
            # 3. Safe search detection
            safe_search_response = self.vision_client.safe_search_detection(image=image)
            logo_analysis['safe_search'] = {
                'adult': safe_search_response.safe_search_annotation.adult.name,
                'violence': safe_search_response.safe_search_annotation.violence.name,
                'racy': safe_search_response.safe_search_annotation.racy.name,
                'spoof': safe_search_response.safe_search_annotation.spoof.name,
                'medical': safe_search_response.safe_search_annotation.medical.name
            }
            
            # 4. Object detection
            objects_response = self.vision_client.object_localization(image=image)
            logo_analysis['object_detection'] = [
                {
                    'name': obj.name,
                    'confidence': obj.score
                }
                for obj in objects_response.localized_object_annotations[:5]
            ]
            
            # 5. Image quality assessment
            logo_analysis['image_quality_score'] = self._assess_image_quality(image_content)
            
            # 6. Calculate overall analysis confidence
            logo_analysis['analysis_confidence'] = self._calculate_logo_analysis_confidence(logo_analysis)
            
            # 7. Processing time
            logo_analysis['processing_time'] = time.time() - start_time
            
            if self.config['enable_detailed_logging']:
                print(f"✅ Logo analysis completed in {logo_analysis['processing_time']:.2f}s")
            
            return logo_analysis
            
        except Exception as e:
            print(f"❌ Logo analysis failed: {e}")
            return {
                'error': str(e),
                'has_logo': False,
                'professional_quality': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _fallback_logo_analysis(self, logo_file):
        """Fallback logo analysis without Vision AI"""
        try:
            if hasattr(logo_file, 'seek'):
                logo_file.seek(0)
            image_content = logo_file.read()
            
            # Basic analysis using PIL
            image = Image.open(io.BytesIO(image_content))
            
            return {
                'has_logo': True,
                'professional_quality': self._basic_image_quality_check(image),
                'image_quality_score': 0.5,  # Default score
                'analysis_confidence': 0.3,  # Lower confidence for fallback
                'fallback_analysis': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'has_logo': False,
                'professional_quality': 0.0,
                'fallback_analysis': True
            }
    
    def _calculate_design_quality(self, properties_response):
        """Calculate logo design quality based on color properties"""
        colors = properties_response.dominant_colors_annotation.colors
        
        # Professional logos typically have:
        # - Limited color palette (2-4 colors)
        # - Good contrast ratios
        # - Balanced color distribution
        
        significant_colors = [c for c in colors if c.score > 0.1]
        color_count = len(significant_colors)
        
        # Optimal range is 2-4 colors
        if 2 <= color_count <= 4:
            quality_score = 0.9
        elif color_count == 1:
            quality_score = 0.7  # Monochrome can be professional
        elif color_count <= 6:
            quality_score = 0.6
        else:
            quality_score = 0.3  # Too many colors might indicate low quality
        
        # Adjust based on color distribution
        if colors:
            max_score = max(color.score for color in colors[:3])
            if max_score > 0.8:  # One dominant color
                quality_score *= 0.9
        
        return quality_score
    
    def _analyze_color_scheme(self, colors):
        """Analyze the color scheme of the logo"""
        color_info = []
        for color in colors[:5]:  # Top 5 colors
            color_info.append({
                'red': color.color.red,
                'green': color.color.green,
                'blue': color.color.blue,
                'alpha': getattr(color.color, 'alpha', 1.0),
                'score': color.score,
                'hex': f"#{color.color.red:02x}{color.color.green:02x}{color.color.blue:02x}"
            })
        
        return {
            'dominant_colors': color_info,
            'color_diversity': len([c for c in color_info if c['score'] > 0.1])
        }
    
    def _assess_image_quality(self, image_content):
        """Assess technical quality of the image"""
        try:
            image = Image.open(io.BytesIO(image_content))
            width, height = image.size
            
            # Quality factors
            quality_score = 0.0
            
            # 1. Resolution assessment
            pixel_count = width * height
            if pixel_count > 250000:  # Very high resolution
                quality_score += 0.4
            elif pixel_count > 100000:  # High resolution
                quality_score += 0.3
            elif pixel_count > 40000:  # Medium resolution
                quality_score += 0.2
            else:  # Low resolution
                quality_score += 0.1
            
            # 2. Aspect ratio assessment (logos usually have reasonable ratios)
            aspect_ratio = max(width, height) / min(width, height)
            if 1.0 <= aspect_ratio <= 3.0:  # Reasonable aspect ratio
                quality_score += 0.2
            elif aspect_ratio <= 5.0:
                quality_score += 0.1
            
            # 3. File size relative to dimensions (indicates compression quality)
            file_size = len(image_content)
            size_per_pixel = file_size / pixel_count
            if size_per_pixel > 0.5:  # Good compression ratio
                quality_score += 0.2
            elif size_per_pixel > 0.2:
                quality_score += 0.1
            
            # 4. Image mode (RGB/RGBA is better than palette mode)
            if image.mode in ['RGB', 'RGBA']:
                quality_score += 0.2
            elif image.mode in ['L', 'LA']:  # Grayscale
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception:
            return 0.3  # Default score if analysis fails
    
    def _basic_image_quality_check(self, image):
        """Basic image quality check without Vision API"""
        try:
            width, height = image.size
            pixel_count = width * height
            
            if pixel_count > 100000 and image.mode in ['RGB', 'RGBA']:
                return 0.7
            elif pixel_count > 40000:
                return 0.5
            else:
                return 0.3
        except Exception:
            return 0.3
    
    def _calculate_logo_analysis_confidence(self, logo_analysis):
        """Calculate overall confidence in logo analysis"""
        confidence_factors = []
        
        # Text detection confidence
        if logo_analysis.get('text_in_logo'):
            avg_text_confidence = np.mean([
                item.get('confidence', 0.9) 
                for item in logo_analysis['text_in_logo']
            ])
            confidence_factors.append(avg_text_confidence)
        
        # Image quality confidence
        if logo_analysis.get('image_quality_score'):
            confidence_factors.append(logo_analysis['image_quality_score'])
        
        # Safe search confidence (inverse of suspicious content)
        safe_search = logo_analysis.get('safe_search', {})
        safe_factors = [safe_search.get(key, 'UNKNOWN') == 'VERY_UNLIKELY' for key in ['adult', 'violence', 'racy']]
        if safe_factors:
            confidence_factors.append(sum(safe_factors) / len(safe_factors))
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    # ========== VERTEX AI TEXT ANALYSIS METHODS ==========
    
    def vertex_ai_text_analysis(self, job_data):
        """Comprehensive text analysis using Vertex AI Natural Language"""
        if not self.vertex_ai_enabled or not self.language_client:
            return self._fallback_text_analysis(job_data)
        
        start_time = time.time()
        
        try:
            # Combine all text fields
            combined_text = self._prepare_text_for_analysis(job_data)
            
            if not combined_text.strip():
                return {
                    'error': 'No text content to analyze',
                    'sentiment_score': 0.0,
                    'ai_generated_probability': 0.5,
                    'flags': ['No content provided']
                }
            
            # Create document for analysis
            document = language_v1.Document(
                content=combined_text,
                type_=language_v1.Document.Type.PLAIN_TEXT
            )
            
            analysis_results = {
                'sentiment_score': 0.0,
                'sentiment_magnitude': 0.0,
                'ai_generated_probability': 0.0,
                'entity_analysis': [],
                'syntax_analysis': {},
                'flags': [],
                'confidence': 0.0,
                'language_detected': 'en',
                'processing_time': 0.0
            }
            
            # 1. Sentiment analysis
            try:
                sentiment_response = self.language_client.analyze_sentiment(
                    request={"document": document}
                )
                analysis_results['sentiment_score'] = sentiment_response.document_sentiment.score
                analysis_results['sentiment_magnitude'] = sentiment_response.document_sentiment.magnitude
            except Exception as e:
                analysis_results['flags'].append(f"Sentiment analysis failed: {str(e)}")
            
            # 2. Entity analysis
            try:
                entities_response = self.language_client.analyze_entities(
                    request={"document": document}
                )
                analysis_results['entity_analysis'] = [
                    {
                        'name': entity.name,
                        'type': entity.type_.name,
                        'salience': entity.salience,
                        'mentions': len(entity.mentions)
                    }
                    for entity in entities_response.entities[:10]
                ]
            except Exception as e:
                analysis_results['flags'].append(f"Entity analysis failed: {str(e)}")
            
            # 3. Syntax analysis
            try:
                syntax_response = self.language_client.analyze_syntax(
                    request={"document": document}
                )
                analysis_results['syntax_analysis'] = self._process_syntax_analysis(syntax_response)
            except Exception as e:
                analysis_results['flags'].append(f"Syntax analysis failed: {str(e)}")
            
            # 4. AI-generated content detection
            analysis_results['ai_generated_probability'] = self._detect_ai_generated_content(combined_text)
            
            # 5. Advanced fraud indicators
            fraud_flags = self._identify_advanced_fraud_patterns(job_data, analysis_results)
            analysis_results['flags'].extend(fraud_flags)
            
            # 6. Calculate overall confidence
            analysis_results['confidence'] = self._calculate_vertex_confidence(analysis_results)
            
            # 7. Processing time
            analysis_results['processing_time'] = time.time() - start_time
            
            if self.config['enable_detailed_logging']:
                print(f"✅ Vertex AI analysis completed in {analysis_results['processing_time']:.2f}s")
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Vertex AI text analysis failed: {e}")
            return {
                'error': str(e),
                'sentiment_score': 0.0,
                'ai_generated_probability': 0.5,
                'flags': [f"Analysis failed: {str(e)}"],
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
    
    def _fallback_text_analysis(self, job_data):
        """Fallback text analysis without Vertex AI"""
        combined_text = self._prepare_text_for_analysis(job_data)
        sentiment_data = self.get_enhanced_sentiment(combined_text)
        
        return {
            'sentiment_score': sentiment_data['polarity'],
            'sentiment_magnitude': abs(sentiment_data['polarity']),
            'ai_generated_probability': self._basic_ai_detection(combined_text),
            'flags': self._basic_fraud_flags(job_data),
            'confidence': 0.6,
            'fallback_analysis': True,
            'word_count': sentiment_data['word_count']
        }
    
    def _prepare_text_for_analysis(self, job_data):
        """Prepare combined text for analysis"""
        text_parts = []
        
        fields_to_analyze = ['title', 'description', 'requirements', 'company_profile', 'benefits']
        weights = {'title': 2, 'description': 3, 'requirements': 2, 'company_profile': 1, 'benefits': 1}
        
        for field in fields_to_analyze:
            content = job_data.get(field, '')
            if content and content.strip():
                # Weight important fields by repetition
                weight = weights.get(field, 1)
                text_parts.extend([content.strip()] * weight)
        
        return ' '.join(text_parts)
    
    def _process_syntax_analysis(self, syntax_response):
        """Process syntax analysis response"""
        tokens = syntax_response.tokens
        
        if not tokens:
            return {}
        
        # Analyze sentence structure
        sentences = {}
        pos_counts = {}
        
        for token in tokens:
            # Part of speech analysis
            pos_tag = token.part_of_speech.tag.name
            pos_counts[pos_tag] = pos_counts.get(pos_tag, 0) + 1
        
        # Calculate sentence complexity metrics
        total_tokens = len(tokens)
        avg_sentence_length = total_tokens / max(len(syntax_response.sentences), 1)
        
        return {
            'total_tokens': total_tokens,
            'avg_sentence_length': avg_sentence_length,
            'pos_distribution': pos_counts,
            'sentence_count': len(syntax_response.sentences)
        }
    
    def _detect_ai_generated_content(self, text):
        """Advanced AI-generated content detection"""
        if not text or len(text.strip()) < 50:
            return 0.5
        
        ai_indicators = 0
        total_indicators = 0
        
        # 1. Check for AI-typical phrases and patterns
        ai_phrases = [
            'leverage', 'utilize', 'optimize', 'streamline', 'synergy', 'paradigm',
            'cutting-edge', 'state-of-the-art', 'innovative solutions', 'best-in-class',
            'world-class', 'industry-leading', 'seamless', 'robust', 'scalable',
            'end-to-end', 'holistic', 'comprehensive solution', 'game-changer'
        ]
        
        text_lower = text.lower()
        ai_phrase_count = sum(1 for phrase in ai_phrases if phrase in text_lower)
        total_indicators += 1
        if ai_phrase_count > 3:
            ai_indicators += min(ai_phrase_count / 10, 1)
        
        # 2. Sentence structure consistency (AI tends to be very consistent)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) >= 3:
            sentence_lengths = [len(s.split()) for s in sentences]
            if sentence_lengths:
                length_std = np.std(sentence_lengths)
                avg_length = np.mean(sentence_lengths)
                
                # Very consistent sentence lengths indicate AI
                consistency_ratio = length_std / max(avg_length, 1)
                if consistency_ratio < 0.3:  # Very consistent
                    ai_indicators += 0.8
                elif consistency_ratio < 0.5:
                    ai_indicators += 0.4
                
                total_indicators += 1
        
        # 3. Vocabulary diversity
        words = text_lower.split()
        if len(words) > 50:
            unique_words = len(set(words))
            vocabulary_diversity = unique_words / len(words)
            
            # AI often has lower vocabulary diversity
            if vocabulary_diversity < 0.4:
                ai_indicators += 0.6
            elif vocabulary_diversity < 0.6:
                ai_indicators += 0.3
            
            total_indicators += 1
        
        # 4. Repetitive patterns
        if len(words) > 50:
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only count significant words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                # High repetition of specific words
                if max_freq > len(words) * 0.05:
                    ai_indicators += 0.7
                elif max_freq > len(words) * 0.03:
                    ai_indicators += 0.4
            
            total_indicators += 1
        
        # 5. Perfect grammar/punctuation (sometimes indicates AI)
        grammar_score = self._assess_grammar_quality(text)
        if grammar_score > 0.95:  # Suspiciously perfect
            ai_indicators += 0.5
        total_indicators += 1
        
        return min(ai_indicators / max(total_indicators, 1), 1.0) if total_indicators > 0 else 0.5
    
    def _assess_grammar_quality(self, text):
        """Basic grammar quality assessment"""
        try:
            # Simple heuristics for grammar quality
            sentences = text.split('.')
            
            quality_score = 0.0
            checks = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 5:
                    checks += 1
                    
                    # Check capitalization
                    if sentence[0].isupper():
                        quality_score += 0.25
                    
                    # Check for basic sentence structure
                    if any(word in sentence.lower() for word in ['the', 'a', 'an', 'is', 'are', 'was', 'were']):
                        quality_score += 0.25
                    
                    # Check for reasonable length
                    if 5 <= len(sentence.split()) <= 30:
                        quality_score += 0.25
                    
                    # Check for ending punctuation
                    if sentence.rstrip().endswith(('.', '!', '?')):
                        quality_score += 0.25
            
            return quality_score / max(checks, 1) if checks > 0 else 0.5
            
        except Exception:
            return 0.7  # Default to good grammar if assessment fails
    
    def _basic_ai_detection(self, text):
        """Basic AI detection without advanced models"""
        if not text:
            return 0.5
        
        ai_indicators = []
        
        # Check for overly generic language
        generic_phrases = ['exciting opportunity', 'dynamic team', 'fast-paced environment', 'competitive salary']
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
        ai_indicators.append(min(generic_count / 5, 1.0))
        
        # Check for perfect grammar (simplified)
        sentences = text.split('.')
        if len(sentences) > 2:
            perfect_caps = sum(1 for s in sentences if s.strip() and s.strip()[0].isupper())
            perfect_ratio = perfect_caps / len([s for s in sentences if s.strip()])
            if perfect_ratio > 0.9:
                ai_indicators.append(0.6)
            else:
                ai_indicators.append(0.3)
        
        return np.mean(ai_indicators) if ai_indicators else 0.4
    
    def _identify_advanced_fraud_patterns(self, job_data, vertex_analysis):
        """Identify advanced fraud patterns using Vertex AI insights"""
        flags = []
        
        # 1. Analyze entities for suspicious patterns
        entities = vertex_analysis.get('entity_analysis', [])
        person_entities = [e for e in entities if e['type'] == 'PERSON']
        organization_entities = [e for e in entities if e['type'] == 'ORGANIZATION']
        
        if not organization_entities and job_data.get('company_profile'):
            flags.append("No organization entities detected despite company information")
        
        # 2. Sentiment-based flags
        sentiment_score = vertex_analysis.get('sentiment_score', 0.0)
        sentiment_magnitude = vertex_analysis.get('sentiment_magnitude', 0.0)
        
        if sentiment_score > 0.8 and sentiment_magnitude > 0.8:
            flags.append("Extremely positive sentiment may indicate exaggerated claims")
        elif sentiment_score < -0.5:
            flags.append("Negative sentiment detected in job description")
        
        # 3. Syntax-based analysis
        syntax_data = vertex_analysis.get('syntax_analysis', {})
        avg_sentence_length = syntax_data.get('avg_sentence_length', 0)
        
        if avg_sentence_length < 5:
            flags.append("Very short sentences may indicate low-quality content")
        elif avg_sentence_length > 30:
            flags.append("Extremely long sentences may indicate AI generation")
        
        # 4. High AI probability flag
        ai_prob = vertex_analysis.get('ai_generated_probability', 0.0)
        if ai_prob > self.config['ai_detection_threshold']:
            flags.append(f"High probability ({ai_prob:.1%}) of AI-generated content")
        
        return flags
    
    def _basic_fraud_flags(self, job_data):
        """Basic fraud flags without advanced analysis"""
        flags = []
        
        # Check for missing critical information
        if not job_data.get('company_profile', '').strip():
            flags.append("Missing company information")
        
        if not job_data.get('requirements', '').strip():
            flags.append("Missing job requirements")
        
        description = job_data.get('description', '')
        if len(description.split()) < 20:
            flags.append("Very short job description")
        
        # Check for suspicious patterns
        if any(word in description.lower() for word in ['easy money', 'work from home guaranteed', 'no experience required']):
            flags.append("Contains suspicious phrases")
        
        return flags
    
    def _calculate_vertex_confidence(self, analysis_results):
        """Calculate confidence score for Vertex AI analysis"""
        confidence_factors = []
        
        # Sentiment confidence (based on magnitude)
        sentiment_magnitude = analysis_results.get('sentiment_magnitude', 0.0)
        confidence_factors.append(min(sentiment_magnitude * 2, 1.0))
        
        # Entity analysis confidence
        entities = analysis_results.get('entity_analysis', [])
        if entities:
            avg_salience = np.mean([e['salience'] for e in entities])
            confidence_factors.append(avg_salience)
        
        # Syntax analysis confidence
        syntax_data = analysis_results.get('syntax_analysis', {})
        if syntax_data.get('total_tokens', 0) > 50:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # AI detection confidence (distance from uncertainty)
        ai_prob = analysis_results.get('ai_generated_probability', 0.5)
        confidence_factors.append(abs(ai_prob - 0.5) * 2)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    # ========== COMPANY VERIFICATION METHODS ==========
    
    def enhanced_company_verification(self, company_data, logo_analysis):
        """Enhanced company verification with multiple data sources"""
        start_time = time.time()
        verification_score = 0.0
        flags = []
        
        try:
            company_name = company_data.get('company_profile', '').strip()
            
            # 1. Logo-company name consistency
            logo_consistency_score = self._verify_logo_consistency(company_name, logo_analysis)
            verification_score += logo_consistency_score * 0.3
            
            if logo_consistency_score < 0.5 and logo_analysis.get('has_logo'):
                flags.append("Company name doesn't match logo text")
            
            # 2. Professional logo quality assessment
            logo_quality = logo_analysis.get('professional_quality', 0.0)
            if logo_quality > 0.7:
                verification_score += 0.25
            elif logo_quality < 0.3 and logo_analysis.get('has_logo'):
                flags.append("Low-quality or suspicious logo")
                verification_score += 0.1
            else:
                verification_score += logo_quality * 0.25
            
            # 3. Email domain verification
            contact_email = company_data.get('contact_email', '')
            email_score = self._verify_email_domain(contact_email)
            verification_score += email_score * 0.2
            
            if email_score < 0.3 and contact_email:
                flags.append("Suspicious email domain")
            
            # 4. Location verification
            location = company_data.get('location', '')
            location_score = self._verify_location(location)
            verification_score += location_score * 0.15
            
            if location_score < 0.3 and location:
                flags.append("Invalid or suspicious location")
            
            # 5. Salary reasonableness
            salary_range = company_data.get('salary_range', '')
            salary_score = self._assess_salary_reasonableness(
                salary_range,
                company_data.get('industry', ''),
                company_data.get('required_experience', '')
            )
            verification_score += salary_score * 0.1
            
            if salary_score < 0.3 and salary_range:
                flags.append("Unrealistic salary range")
            
            # 6. Content consistency check
            consistency_score = self._check_content_consistency(company_data)
            verification_score += consistency_score * 0.1
            
            if consistency_score < 0.3:
                flags.append("Inconsistent job posting content")
            
            # Normalize score
            verification_score = min(1.0, verification_score)
            
            # Determine risk level
            if verification_score >= 0.7:
                risk_level = 'low'
            elif verification_score >= 0.4:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            result = {
                'verification_score': verification_score,
                'flags': flags,
                'risk_level': risk_level,
                'processing_time': time.time() - start_time,
                'components': {
                    'logo_consistency': logo_consistency_score,
                    'logo_quality': logo_quality,
                    'email_domain': email_score,
                    'location_validity': location_score,
                    'salary_reasonableness': salary_score,
                    'content_consistency': consistency_score
                }
            }
            
            if self.config['enable_detailed_logging']:
                print(f"✅ Company verification completed: {verification_score:.2f} ({risk_level} risk)")
            
            return result
            
        except Exception as e:
            return {
                'verification_score': 0.0,
                'flags': [f"Verification failed: {str(e)}"],
                'risk_level': 'high',
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _verify_logo_consistency(self, company_name, logo_analysis):
        """Verify consistency between company name and logo text"""
        if not company_name or not logo_analysis.get('text_in_logo'):
            return 0.5  # Neutral score if no data
        
        try:
            company_words = set(company_name.lower().split())
            logo_texts = logo_analysis['text_in_logo']
            
            if isinstance(logo_texts[0], dict):
                logo_text = ' '.join([item['text'] for item in logo_texts])
            else:
                logo_text = ' '.join(logo_texts)
            
            logo_words = set(logo_text.lower().split())
            
            # Calculate word overlap
            common_words = company_words.intersection(logo_words)
            if not company_words:
                return 0.5
            
            overlap_ratio = len(common_words) / len(company_words)
            
            # Boost score for exact matches
            if company_name.lower() in logo_text.lower():
                overlap_ratio = min(overlap_ratio + 0.3, 1.0)
            
            return overlap_ratio
            
        except Exception:
            return 0.5
    
    def _verify_email_domain(self, email):
        """Enhanced email domain verification"""
        if not email or '@' not in email:
            return 0.0
        
        try:
            domain = email.split('@')[1].lower()
            
            # Common legitimate domains
            legitimate_domains = {
                'gmail.com': 0.7, 'yahoo.com': 0.7, 'outlook.com': 0.7, 'hotmail.com': 0.6,
                'aol.com': 0.6, 'icloud.com': 0.7, 'protonmail.com': 0.8
            }
            
            # Business-like domains
            if domain in legitimate_domains:
                return legitimate_domains[domain]
            
            # Suspicious patterns
            suspicious_patterns = [
                'temp', 'fake', 'test', '10min', 'guerilla', 'mailinator',
                'yopmail', 'throwaway', 'discard', 'spam'
            ]
            
            if any(pattern in domain for pattern in suspicious_patterns):
                return 0.1
            
            # Check domain structure
            domain_parts = domain.split('.')
            if len(domain_parts) < 2:
                return 0.2
            
            # Too many subdomains might be suspicious
            if len(domain_parts) > 4:
                return 0.3
            
            # Check for reasonable TLD
            tld = domain_parts[-1]
            common_tlds = {'com', 'org', 'net', 'edu', 'gov', 'mil', 'int', 'co', 'io', 'ai'}
            country_tlds = {'uk', 'de', 'fr', 'jp', 'ca', 'au', 'in', 'br', 'mx', 'es', 'it', 'ru', 'cn'}
            
            if tld in common_tlds:
                return 0.8
            elif tld in country_tlds:
                return 0.7
            elif len(tld) == 2:  # Other country codes
                return 0.6
            else:
                return 0.4
            
        except Exception:
            return 0.3
    
    def _verify_location(self, location):
        """Enhanced location verification"""
        if not location or not location.strip():
            return 0.0
        
        try:
            location_lower = location.lower().strip()
            
            # Check for obviously fake locations
            fake_indicators = [
                'fake', 'test', 'nowhere', 'invalid', 'n/a', 'tbd',
                'unknown', 'none', 'null', 'undefined'
            ]
            
            if any(indicator in location_lower for indicator in fake_indicators):
                return 0.1
            
            # Check for proper location format
            location_patterns = [
                r'^[A-Za-z\s]+,\s*[A-Za-z\s]+$',  # City, State/Country
                r'^[A-Za-z\s]+-[A-Za-z\s]+$',      # City-State
                r'^[A-Za-z\s]+\s+[A-Z]{2}$',       # City State (with state abbreviation)
            ]
            
            for pattern in location_patterns:
                if re.match(pattern, location):
                    return 0.9
            
            # Check for reasonable content
            words = location.split()
            if len(words) >= 2:
                # Has multiple words, probably legitimate
                return 0.7
            elif len(words) == 1 and len(words[0]) > 2:
                # Single word location (city name)
                return 0.6
            else:
                return 0.3
                
        except Exception:
            return 0.4
    
    def _assess_salary_reasonableness(self, salary_range, industry, experience):
        """Enhanced salary reasonableness assessment"""
        if not salary_range or not salary_range.strip():
            return 0.5  # Neutral if no salary provided
        
        try:
            # Extract numeric values from salary range
            import re
            numbers = re.findall(r'[\d,]+', salary_range.replace(',', ''))
            
            if not numbers:
                return 0.3
            
            # Convert to integers
            salaries = []
            for num in numbers:
                try:
                    salaries.append(int(num.replace(',', '')))
                except ValueError:
                    continue
            
            if not salaries:
                return 0.3
            
            min_salary = min(salaries)
            max_salary = max(salaries)
            
            # Basic reasonableness checks
            score = 0.5
            
            # Check for reasonable ranges
            if max_salary > 2000000:  # Over $2M is very suspicious
                return 0.1
            elif max_salary > 1000000:  # Over $1M might be suspicious depending on role
                score -= 0.3
            elif max_salary > 500000:  # High but possible
                score -= 0.1
            
            if min_salary < 15000:  # Under $15K is very suspicious
                return 0.2
            elif min_salary < 25000:  # Under $25K might be suspicious
                score -= 0.2
            
            # Check salary range consistency
            if len(salaries) >= 2:
                if max_salary > min_salary * 5:  # Range too wide
                    score -= 0.3
                elif max_salary > min_salary * 2.5:  # Range quite wide
                    score -= 0.1
            
            # Industry-specific adjustments (basic)
            if industry:
                industry_lower = industry.lower()
                if any(term in industry_lower for term in ['tech', 'software', 'engineering']):
                    if min_salary < 40000:  # Tech salaries usually higher
                        score -= 0.2
                elif any(term in industry_lower for term in ['retail', 'service', 'hospitality']):
                    if min_salary > 150000:  # Might be high for these industries
                        score -= 0.1
            
            # Experience-based adjustments
            if experience:
                experience_lower = experience.lower()
                if 'entry' in experience_lower or 'junior' in experience_lower:
                    if min_salary > 100000:  # High for entry level
                        score -= 0.2
                elif 'senior' in experience_lower or 'lead' in experience_lower:
                    if max_salary < 60000:  # Low for senior roles
                        score -= 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.4
    
    def _check_content_consistency(self, company_data):
        """Check consistency across different fields"""
        try:
            score = 1.0
            
            # Check title-description consistency
            title = company_data.get('title', '').lower()
            description = company_data.get('description', '').lower()
            
            if title and description:
                title_words = set(title.split())
                description_words = set(description.split())
                
                # Some title words should appear in description
                overlap = title_words.intersection(description_words)
                if len(title_words) > 0:
                    overlap_ratio = len(overlap) / len(title_words)
                    if overlap_ratio < 0.3:  # Very little overlap
                        score -= 0.3
            
            # Check company-description consistency
            company_profile = company_data.get('company_profile', '').lower()
            if company_profile and description:
                # Company name should be mentioned somewhere
                company_words = company_profile.split()[:3]  # First few words likely company name
                if company_words and not any(word in description for word in company_words):
                    score -= 0.2
            
            # Check salary-level consistency
            salary_range = company_data.get('salary_range', '')
            required_experience = company_data.get('required_experience', '').lower()
            
            if salary_range and required_experience:
                # Basic check: entry level shouldn't have very high salary
                if 'entry' in required_experience and any(char.isdigit() for char in salary_range):
                    numbers = re.findall(r'\d+', salary_range.replace(',', ''))
                    if numbers and int(numbers[-1]) > 100000:
                        score -= 0.2
            
            return max(0.0, score)
            
        except Exception:
            return 0.5
    
    # ========== MULTI-MODAL PREDICTION METHODS ==========
    
    def multi_modal_fraud_prediction(self, text_data, logo_analysis, vertex_analysis, company_verification):
        """Comprehensive multi-modal fraud prediction"""
        start_time = time.time()
        
        try:
            # Calculate individual risk components
            text_risk = self._calculate_text_risk(text_data)
            logo_risk = self._calculate_logo_risk(logo_analysis)
            ai_content_risk = vertex_analysis.get('ai_generated_probability', 0.5)
            company_risk = 1.0 - company_verification.get('verification_score', 0.5)
            sentiment_risk = self._calculate_sentiment_risk(vertex_analysis.get('sentiment_score', 0.0))
            
            # Apply configured weights
            multi_modal_score = (
                text_risk * self.risk_weights['text'] +
                logo_risk * self.risk_weights['logo'] +
                ai_content_risk * self.risk_weights['ai_content'] +
                company_risk * self.risk_weights['company'] +
                sentiment_risk * self.risk_weights['sentiment']
            )
            
            # Determine fraud prediction
            threshold = self.config['fraud_probability_threshold']
            is_fraudulent = multi_modal_score > threshold
            
            # Calculate confidence based on agreement between methods
            confidence = self._calculate_ensemble_confidence([
                text_risk, logo_risk, ai_content_risk, company_risk, sentiment_risk
            ])
            
            # Determine risk level
            risk_level = self._determine_risk_level(multi_modal_score)
            
            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(
                text_risk, logo_risk, ai_content_risk, company_risk, sentiment_risk
            )
            
            result = {
                'is_fraudulent': is_fraudulent,
                'confidence': confidence,
                'fraud_probability': multi_modal_score,
                'multi_modal_score': multi_modal_score,
                'risk_level': risk_level,
                'sentiment_score': vertex_analysis.get('sentiment_score', 0.0),
                'processing_time': time.time() - start_time,
                'risk_breakdown': {
                    'text_risk': text_risk,
                    'logo_risk': logo_risk,
                    'ai_content_risk': ai_content_risk,
                    'company_risk': company_risk,
                    'sentiment_risk': sentiment_risk
                },
                'contributing_factors': contributing_factors,
                'method': 'Multi-Modal Enhanced' if self.vertex_ai_enabled else 'Multi-Modal Basic'
            }
            
            if self.config['enable_detailed_logging']:
                print(f"✅ Multi-modal prediction: {multi_modal_score:.1%} fraud probability ({risk_level} risk)")
            
            return result
            
        except Exception as e:
            print(f"❌ Multi-modal prediction failed: {e}")
            # Fallback to basic prediction
            return self._fallback_prediction(text_data, e)
    
    def _calculate_text_risk(self, text_data):
        """Calculate fraud risk based on text content"""
        if self.is_trained and self.model:
            try:
                # Use trained ML model
                features = self.extract_features(text_data)
                combined_text = ' '.join([
                    text_data.get('title', ''),
                    text_data.get('description', ''),
                    text_data.get('requirements', ''),
                    text_data.get('company_profile', '')
                ])
                cleaned_text = self.clean_text(combined_text)
                
                text_features = self.vectorizer.transform([cleaned_text])
                numerical_values = [features[name] for name in self.feature_names]
                X = np.hstack([text_features.toarray(), [numerical_values]])
                
                probabilities = self.model.predict_proba(X)[0]
                return probabilities if len(probabilities) > 1 else probabilities
                
            except Exception as e:
                if self.config['enable_detailed_logging']:
                    print(f"ML text risk calculation failed: {e}")
                return self._rule_based_text_risk(text_data)
        else:
            return self._rule_based_text_risk(text_data)
    
    def _rule_based_text_risk(self, text_data):
        """Rule-based text risk calculation"""
        features = self.extract_features(text_data)
        fraud_probability, _ = self.rule_based_prediction(features)
        return fraud_probability
    
    def _calculate_logo_risk(self, logo_analysis):
        """Calculate risk based on logo analysis"""
        if not logo_analysis or logo_analysis.get('error'):
            return 0.5  # Neutral risk if no logo or analysis failed
        
        risk_score = 0.0
        
        # No logo when claimed to have one
        if not logo_analysis.get('has_logo', False):
            risk_score += 0.3
        
        # Low professional quality
        professional_quality = logo_analysis.get('professional_quality', 0.5)
        risk_score += (1.0 - professional_quality) * 0.4
        
        # Suspicious safe search results
        safe_search = logo_analysis.get('safe_search', {})
        suspicious_content = any(
            safe_search.get(key, 'UNKNOWN') not in ['VERY_UNLIKELY', 'UNLIKELY']
            for key in ['adult', 'violence', 'racy']
        )
        if suspicious_content:
            risk_score += 0.3
        
        # Low image quality
        image_quality = logo_analysis.get('image_quality_score', 0.5)
        if image_quality < 0.3:
            risk_score += 0.2
        
        # Low analysis confidence
        analysis_confidence = logo_analysis.get('analysis_confidence', 0.5)
        if analysis_confidence < 0.5:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _calculate_sentiment_risk(self, sentiment_score):
        """Calculate risk based on sentiment analysis"""
        if sentiment_score is None:
            return 0.3
        
        # Extremely positive sentiment might indicate fraud
        if sentiment_score > 0.8:
            return 0.7
        # Negative sentiment is also suspicious
        elif sentiment_score < -0.5:
            return 0.6
        # Very neutral might be AI-generated
        elif abs(sentiment_score) < 0.1:
            return 0.4
        else:
            return abs(sentiment_score) * 0.3
    
    def _calculate_ensemble_confidence(self, risk_scores):
        """Calculate confidence based on agreement between different methods"""
        if not risk_scores or len(risk_scores) < 2:
            return 0.5
        
        # Calculate variance - lower variance means higher agreement
        variance = np.var(risk_scores)
        
        # Calculate average distance from mean
        mean_score = np.mean(risk_scores)
        avg_deviation = np.mean([abs(score - mean_score) for score in risk_scores])
        
        # High agreement = high confidence
        agreement_confidence = max(0.3, 1.0 - (variance * 4))  # Scale variance
        deviation_confidence = max(0.3, 1.0 - (avg_deviation * 2))  # Scale deviation
        
        # Confidence is higher when we're further from uncertainty (0.5)
        certainty_confidence = abs(mean_score - 0.5) * 2
        
        # Combine confidence metrics
        overall_confidence = (agreement_confidence + deviation_confidence + certainty_confidence) / 3
        
        return min(0.95, max(0.5, overall_confidence))
    
    def _determine_risk_level(self, fraud_probability):
        """Determine risk level based on fraud probability"""
        if fraud_probability >= 0.85:
            return 'CRITICAL'
        elif fraud_probability >= 0.7:
            return 'HIGH'
        elif fraud_probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _identify_contributing_factors(self, text_risk, logo_risk, ai_content_risk, company_risk, sentiment_risk):
        """Identify main factors contributing to fraud prediction"""
        factors = {
            'text_analysis': text_risk,
            'logo_quality': logo_risk,
            'ai_generated_content': ai_content_risk,
            'company_verification': company_risk,
            'sentiment_analysis': sentiment_risk
        }
        
        # Sort by risk level
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize impact levels
        result = []
        for factor, risk in sorted_factors:
            if risk >= 0.7:
                impact = 'high'
            elif risk >= 0.5:
                impact = 'medium'
            else:
                impact = 'low'
            
            result.append({
                'factor': factor,
                'risk_level': risk,
                'impact': impact,
                'description': self._get_factor_description(factor, risk)
            })
        
        return result
    
    def _get_factor_description(self, factor, risk_level):
        """Get human-readable description of risk factors"""
        descriptions = {
            'text_analysis': {
                'high': 'Text content shows strong fraud indicators',
                'medium': 'Text content has some suspicious characteristics',
                'low': 'Text content appears legitimate'
            },
            'logo_quality': {
                'high': 'Logo quality is poor or missing entirely',
                'medium': 'Logo quality has some concerns',
                'low': 'Logo appears professional and legitimate'
            },
            'ai_generated_content': {
                'high': 'Content appears to be AI-generated',
                'medium': 'Content may be partially AI-generated',
                'low': 'Content appears to be human-written'
            },
            'company_verification': {
                'high': 'Company information failed verification checks',
                'medium': 'Company information has some inconsistencies',
                'low': 'Company information appears verified'
            },
            'sentiment_analysis': {
                'high': 'Sentiment analysis indicates suspicious patterns',
                'medium': 'Sentiment analysis shows some concerns',
                'low': 'Sentiment analysis appears normal'
            }
        }
        
        impact_level = 'high' if risk_level >= 0.7 else 'medium' if risk_level >= 0.5 else 'low'
        return descriptions.get(factor, {}).get(impact_level, f'{factor} shows {impact_level} risk')
    
    def _fallback_prediction(self, text_data, error):
        """Fallback prediction when multi-modal fails"""
        try:
            features = self.extract_features(text_data)
            fraud_probability, confidence = self.rule_based_prediction(features)
            
            return {
                'is_fraudulent': fraud_probability > 0.5,
                'confidence': confidence,
                'fraud_probability': fraud_probability,
                'multi_modal_score': fraud_probability,
                'risk_level': 'HIGH' if fraud_probability > 0.7 else 'MEDIUM' if fraud_probability > 0.4 else 'LOW',
                'sentiment_score': features['desc_sentiment'],
                'method': 'Fallback Rule-based',
                'error': str(error)
            }
        except Exception as fallback_error:
            return {
                'is_fraudulent': True,  # Safe default
                'confidence': 0.5,
                'fraud_probability': 0.7,
                'multi_modal_score': 0.7,
                'risk_level': 'HIGH',
                'sentiment_score': 0.0,
                'method': 'Error Fallback',
                'error': f"Original: {str(error)}, Fallback: {str(fallback_error)}"
            }
    
    # ========== UTILITY METHODS ==========
    
    def extract_features(self, job_data):
        """Enhanced feature extraction with additional metrics"""
        # Get enhanced sentiment data
        desc_sentiment = self.get_enhanced_sentiment(job_data.get('description', ''))
        req_sentiment = self.get_enhanced_sentiment(job_data.get('requirements', ''))
        comp_sentiment = self.get_enhanced_sentiment(job_data.get('company_profile', ''))
        
        # Basic text metrics
        title = job_data.get('title', '')
        description = job_data.get('description', '')
        requirements = job_data.get('requirements', '')
        
        features = {
            # Enhanced sentiment features
            'desc_sentiment': desc_sentiment['polarity'],
            'desc_subjectivity': desc_sentiment['subjectivity'],
            'req_sentiment': req_sentiment['polarity'],
            'req_subjectivity': req_sentiment['subjectivity'],
            'company_sentiment': comp_sentiment['polarity'],
            'company_subjectivity': comp_sentiment['subjectivity'],
            
            # Text length and complexity features
            'title_length': len(title) if title else 0,
            'description_length': len(description) if description else 0,
            'requirements_length': len(requirements) if requirements else 0,
            'title_word_count': desc_sentiment['word_count'] if 'word_count' in desc_sentiment else len(title.split()),
            'description_word_count': desc_sentiment['word_count'],
            'avg_word_length': desc_sentiment['avg_word_length'],
            
            # Missing information flags
            'missing_salary': 1 if not job_data.get('salary_range', '').strip() else 0,
            'missing_company': 1 if not job_data.get('company_profile', '').strip() else 0,
            'missing_requirements': 1 if not job_data.get('requirements', '').strip() else 0,
            
            # Binary features
            'has_company_logo': 1 if job_data.get('has_company_logo') else 0,
            'has_questions': 1 if job_data.get('has_questions') else 0,
            'telecommuting': 1 if job_data.get('telecommuting') else 0,
        }
        
        return features
    
    def rule_based_prediction(self, features):
        """Enhanced rule-based fraud detection"""
        fraud_score = 0.0
        confidence = 0.8
        
        # Weight-based scoring system
        weights = {
            'missing_company': 0.35,
            'missing_salary': 0.25,
            'missing_requirements': 0.15,
            'short_description': 0.30,
            'very_short_description': 0.45,
            'short_title': 0.20,
            'negative_sentiment': 0.25,
            'very_negative_sentiment': 0.40,
            'no_logo': 0.15,
            'no_questions': 0.10,
            'overly_positive': 0.20,
            'high_subjectivity': 0.15
        }
        
        # Apply rules with weights
        if features['missing_company']:
            fraud_score += weights['missing_company']
        
        if features['missing_salary']:
            fraud_score += weights['missing_salary']
        
        if features['missing_requirements']:
            fraud_score += weights['missing_requirements']
        
        # Description length analysis
        desc_length = features['description_length']
        if desc_length < 50:
            fraud_score += weights['very_short_description']
        elif desc_length < 100:
            fraud_score += weights['short_description']
        
        # Title length analysis
        if features['title_length'] < 10:
            fraud_score += weights['short_title']
        
        # Sentiment analysis
        desc_sentiment = features['desc_sentiment']
        if desc_sentiment < -0.5:
            fraud_score += weights['very_negative_sentiment']
        elif desc_sentiment < -0.2:
            fraud_score += weights['negative_sentiment']
        elif desc_sentiment > 0.8:
            fraud_score += weights['overly_positive']
        
        # Subjectivity analysis
        if features['desc_subjectivity'] > 0.8:
            fraud_score += weights['high_subjectivity']
        
        # Company indicators
        if not features['has_company_logo']:
            fraud_score += weights['no_logo']
        
        if not features['has_questions']:
            fraud_score += weights['no_questions']
        
        # Normalize and cap the score
        fraud_probability = min(fraud_score, 1.0)
        
        return fraud_probability, confidence
    
    def process_bulk_analysis(self, csv_file):
        """Process bulk job postings from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            results = []
            
            print(f"🔄 Processing {len(df)} job postings...")
            
            for index, row in df.iterrows():
                try:
                    job_data = row.to_dict()
                    
                    # Clean data
                    for key, value in job_data.items():
                        if pd.isna(value):
                            job_data[key] = ''
                    
                    # Run multi-modal analysis (without logo for bulk processing)
                    vertex_analysis = self.vertex_ai_text_analysis(job_data) if self.vertex_ai_enabled else self._fallback_text_analysis(job_data)
                    company_verification = self.enhanced_company_verification(job_data, {})
                    
                    prediction = self.multi_modal_fraud_prediction(
                        job_data, {}, vertex_analysis, company_verification
                    )
                    
                    results.append({
                        'row_index': index + 1,
                        'title': job_data.get('title', 'N/A')[:50],
                        'company': job_data.get('company_profile', 'N/A')[:50],
                        'is_fraudulent': prediction['is_fraudulent'],
                        'fraud_probability': prediction['fraud_probability'],
                        'risk_level': prediction['risk_level'],
                        'confidence': prediction['confidence'],
                        'flags': vertex_analysis.get('flags', [])[:3],  # Limit flags
                        'processing_time': prediction.get('processing_time', 0.0)
                    })
                    
                    if (index + 1) % 10 == 0:
                        print(f"   Processed {index + 1}/{len(df)} jobs...")
                        
                except Exception as e:
                    print(f"   Error processing row {index + 1}: {e}")
                    results.append({
                        'row_index': index + 1,
                        'title': 'ERROR',
                        'company': 'ERROR',
                        'is_fraudulent': True,
                        'fraud_probability': 1.0,
                        'risk_level': 'HIGH',
                        'error': str(e)
                    })
            
            print(f"✅ Bulk processing completed: {len(results)} results")
            return results
            
        except Exception as e:
            print(f"❌ Bulk processing failed: {e}")
            raise Exception(f"Bulk processing failed: {str(e)}")
    
    # Main prediction interface (backward compatibility)
    def predict_fraud(self, job_data):
        """Main prediction function with enhanced capabilities"""
        try:
            start_time = time.time()
            
            # Run enhanced analysis if Vertex AI is available
            if self.vertex_ai_enabled:
                # Vertex AI text analysis
                vertex_analysis = self.vertex_ai_text_analysis(job_data)
                
                # Company verification (without logo for basic prediction)
                company_verification = self.enhanced_company_verification(job_data, {})
                
                # Multi-modal prediction
                result = self.multi_modal_fraud_prediction(
                    job_data, {}, vertex_analysis, company_verification
                )
                
            else:
                # Fallback to traditional method
                features = self.extract_features(job_data)
                
                if self.is_trained and self.model:
                    fraud_probability, confidence = self.ml_prediction(job_data)
                    method = "Traditional ML"
                else:
                    fraud_probability, confidence = self.rule_based_prediction(features)
                    method = "Enhanced Rule-based"
                
                # Determine risk level and create result
                is_fraudulent = fraud_probability > 0.5
                
                if fraud_probability >= 0.8:
                    risk_level = 'CRITICAL'
                elif fraud_probability >= 0.7:
                    risk_level = 'HIGH'
                elif fraud_probability >= 0.4:
                    risk_level = 'MEDIUM'
                else:
                    risk_level = 'LOW'
                
                result = {
                    'is_fraudulent': is_fraudulent,
                    'confidence': confidence,
                    'fraud_probability': fraud_probability,
                    'sentiment_score': features['desc_sentiment'],
                    'risk_level': risk_level,
                    'method': method,
                    'features': features
                }
            
            result['processing_time'] = time.time() - start_time
            
            if self.config['enable_detailed_logging']:
                method = result.get('method', 'Unknown')
                fraud_prob = result['fraud_probability']
                risk_level = result['risk_level']
                print(f"🔍 Fraud prediction completed using {method}")
                print(f"   Fraud Probability: {fraud_prob:.1%}, Risk Level: {risk_level}")
            
            return result
            
        except Exception as e:
            print(f"❌ Enhanced prediction error: {e}")
            # Final fallback
            return {
                'is_fraudulent': False,
                'confidence': 0.5,
                'fraud_probability': 0.3,
                'sentiment_score': 0.0,
                'risk_level': 'LOW',
                'method': 'Error fallback',
                'error': str(e),
                'processing_time': 0.0
            }


# Maintain backward compatibility
FraudDetectionService = EnhancedFraudDetectionService
