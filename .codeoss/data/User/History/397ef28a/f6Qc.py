from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import User
from django.utils import timezone
from django.core.files.storage import default_storage
import uuid
import os


def company_logo_upload_path(instance, filename):
    """Generate upload path for company logos"""
    ext = filename.split('.')[-1]
    filename = f"{instance.job_id}_{uuid.uuid4().hex[:8]}.{ext}"
    return f'company_logos/{timezone.now().year}/{timezone.now().month}/{filename}'


class JobPosting(models.Model):
    # Core job posting fields
    job_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=200)
    location = models.CharField(max_length=200, blank=True, null=True)
    department = models.CharField(max_length=100, blank=True, null=True)
    salary_range = models.CharField(max_length=100, blank=True, null=True)
    company_profile = models.TextField(blank=True, null=True)
    description = models.TextField()
    requirements = models.TextField(blank=True, null=True)
    benefits = models.TextField(blank=True, null=True)
    telecommuting = models.BooleanField(default=False)
    has_company_logo = models.BooleanField(default=False)
    has_questions = models.BooleanField(default=False)
    employment_type = models.CharField(max_length=50, blank=True, null=True)
    required_experience = models.CharField(max_length=50, blank=True, null=True)
    required_education = models.CharField(max_length=50, blank=True, null=True)
    industry = models.CharField(max_length=100, blank=True, null=True)
    function = models.CharField(max_length=100, blank=True, null=True)
    
    # Enhanced fields for ML learning and tracking
    DATA_SOURCE_CHOICES = [
        ('CSV', 'Original CSV Dataset'),
        ('USER', 'User Submitted'),
        ('API', 'API Submission'),
        ('BULK', 'Bulk Upload'),
        ('SCRAPER', 'Web Scraper')
    ]
    
    data_source = models.CharField(
        max_length=20, 
        choices=DATA_SOURCE_CHOICES, 
        default='USER'
    )
    
    # User interaction tracking
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    poster_ip = models.GenericIPAddressField(null=True, blank=True)  # Alias for backward compatibility
    user_agent = models.TextField(blank=True, null=True)
    session_id = models.CharField(max_length=100, blank=True, null=True)
    
    # Content analysis metadata
    word_count = models.PositiveIntegerField(null=True, blank=True)
    character_count = models.PositiveIntegerField(null=True, blank=True)
    url_count = models.PositiveIntegerField(default=0)
    email_count = models.PositiveIntegerField(default=0)
    phone_count = models.PositiveIntegerField(default=0)
    
    # Quality indicators
    has_company_website = models.BooleanField(default=False)
    has_contact_info = models.BooleanField(default=False)
    has_specific_requirements = models.BooleanField(default=False)
    description_quality_score = models.FloatField(
        null=True, 
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # === NEW ENHANCED FIELDS FOR VERTEX AI & LOGO ANALYSIS ===
    
    # Logo and Visual Analysis
    company_logo = models.ImageField(
        upload_to=company_logo_upload_path,
        null=True,
        blank=True,
        help_text="Company logo for visual analysis"
    )
    logo_file_size = models.PositiveIntegerField(null=True, blank=True, help_text="Logo file size in bytes")
    logo_dimensions = models.CharField(max_length=20, blank=True, help_text="Logo dimensions (WxH)")
    logo_analysis_result = models.JSONField(
        null=True,
        blank=True,
        help_text="Results from Vision AI logo analysis"
    )
    
    # Vertex AI Analysis Results
    vertex_ai_analysis = models.JSONField(
        null=True,
        blank=True,
        help_text="Complete Vertex AI analysis results"
    )
    
    # AI Content Detection
    ai_generated_probability = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Probability that content is AI-generated"
    )
    ai_detection_confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Enhanced Company Verification
    company_verification_status = models.CharField(
        max_length=20,
        choices=[
            ('PENDING', 'Verification Pending'),
            ('VERIFIED', 'Company Verified'),
            ('SUSPICIOUS', 'Potentially Suspicious'),
            ('FLAGGED', 'Flagged for Review'),
            ('FAILED', 'Verification Failed')
        ],
        default='PENDING'
    )
    company_verification_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    verification_flags = models.JSONField(
        default=list,
        blank=True,
        help_text="List of verification issues found"
    )
    
    # Contact Information Analysis
    contact_email = models.EmailField(blank=True, help_text="Primary contact email")
    contact_phone = models.CharField(max_length=50, blank=True, help_text="Contact phone number")
    contact_website = models.URLField(blank=True, help_text="Company website")
    email_domain_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Email domain legitimacy score"
    )
    
    # Geolocation and IP Analysis
    ip_geolocation = models.JSONField(
        null=True,
        blank=True,
        help_text="IP geolocation data"
    )
    ip_reputation_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    posting_timezone = models.CharField(max_length=50, blank=True)
    
    # Content Analysis Enhancements
    language_detected = models.CharField(max_length=10, default='en')
    readability_score = models.FloatField(null=True, blank=True)
    complexity_score = models.FloatField(null=True, blank=True)
    
    # Behavioral Analysis
    posting_speed = models.FloatField(
        null=True,
        blank=True,
        help_text="Time taken to create posting (seconds)"
    )
    device_fingerprint = models.TextField(blank=True)
    browser_features = models.JSONField(default=dict, blank=True)
    
    # Processing Metadata
    vertex_ai_processing_time = models.FloatField(null=True, blank=True)
    logo_analysis_processing_time = models.FloatField(null=True, blank=True)
    total_analysis_time = models.FloatField(null=True, blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Soft delete
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'job_postings'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['data_source']),
            models.Index(fields=['created_at']),
            models.Index(fields=['industry']),
            models.Index(fields=['employment_type']),
            models.Index(fields=['-created_at', 'data_source']),
            # New indexes for enhanced fields
            models.Index(fields=['company_verification_status']),
            models.Index(fields=['ai_generated_probability']),
            models.Index(fields=['has_company_logo']),
            models.Index(fields=['ip_address']),
        ]

    def __str__(self):
        return f"#{self.job_id} - {self.title}"
    
    def soft_delete(self):
        """Soft delete the job posting"""
        self.is_deleted = True
        self.deleted_at = timezone.now()
        self.save()
    
    def delete_logo(self):
        """Delete associated logo file"""
        if self.company_logo:
            if default_storage.exists(self.company_logo.name):
                default_storage.delete(self.company_logo.name)
            self.company_logo = None
            self.save()
    
    @property
    def has_enhanced_analysis(self):
        """Check if job has been processed with enhanced analysis"""
        return bool(self.vertex_ai_analysis or self.logo_analysis_result)
    
    @property
    def verification_status_display(self):
        """Human-readable verification status"""
        status_map = {
            'VERIFIED': '✅ Verified',
            'SUSPICIOUS': '⚠️ Suspicious',
            'FLAGGED': '🚩 Flagged',
            'FAILED': '❌ Failed',
            'PENDING': '⏳ Pending'
        }
        return status_map.get(self.company_verification_status, self.company_verification_status)


class FraudPrediction(models.Model):
    RISK_LEVELS = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'),
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical'),  # New risk level
    ]
    
    METHOD_CHOICES = [
        ('RULE_BASED', 'Rule-based Algorithm'),
        ('ML_MODEL', 'Machine Learning Model'),
        ('ENSEMBLE', 'Ensemble Method'),
        ('DEEP_LEARNING', 'Deep Learning Model'),
        ('HYBRID', 'Hybrid Approach'),
        ('MULTI_MODAL', 'Multi-Modal Analysis'),  # New method
        ('VERTEX_AI', 'Vertex AI Enhanced'),  # New method
    ]
    
    job_posting = models.OneToOneField(
        JobPosting, 
        on_delete=models.CASCADE,
        related_name='fraud_prediction'
    )
    
    # Core prediction results
    is_fraudulent = models.BooleanField()
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    fraud_probability = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    sentiment_score = models.FloatField(
        validators=[MinValueValidator(-1.0), MaxValueValidator(1.0)]
    )
    risk_level = models.CharField(
        max_length=10, 
        choices=RISK_LEVELS,
        default='LOW'
    )
    
    # Enhanced ML tracking fields
    model_version = models.CharField(max_length=50, default='v1.0')
    prediction_method = models.CharField(
        max_length=20,
        choices=METHOD_CHOICES,
        default='RULE_BASED'
    )
    processing_time = models.FloatField(
        null=True, 
        blank=True,
        help_text="Processing time in seconds"
    )
    
    # Feature importance scores (stored as JSON)
    feature_scores = models.JSONField(default=dict, blank=True)
    
    # Model confidence breakdown
    text_confidence = models.FloatField(null=True, blank=True)
    metadata_confidence = models.FloatField(null=True, blank=True)
    sentiment_confidence = models.FloatField(null=True, blank=True)
    
    # === NEW ENHANCED FIELDS FOR MULTI-MODAL ANALYSIS ===
    
    # Multi-Modal Analysis Results
    multi_modal_risk_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Combined risk score from all analysis methods"
    )
    
    # Individual Analysis Component Scores
    logo_verification_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    vertex_ai_confidence = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    company_verification_flags = models.JSONField(
        default=list,
        blank=True,
        help_text="Issues found during company verification"
    )
    ai_generated_probability = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Risk Breakdown by Component
    risk_breakdown = models.JSONField(
        default=dict,
        blank=True,
        help_text="Detailed breakdown of risk factors"
    )
    contributing_factors = models.JSONField(
        default=list,
        blank=True,
        help_text="Main factors contributing to fraud prediction"
    )
    
    # Logo Analysis Results
    logo_quality_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    logo_brand_match = models.BooleanField(null=True, blank=True)
    logo_professional_quality = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Company Verification Details
    email_domain_legitimacy = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    location_verification_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    contact_verification_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Behavioral Analysis
    posting_pattern_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    ip_reputation_impact = models.FloatField(null=True, blank=True)
    
    # Prediction quality metrics
    entropy = models.FloatField(null=True, blank=True)
    prediction_variance = models.FloatField(null=True, blank=True)
    model_uncertainty = models.FloatField(null=True, blank=True)
    
    # Alert Configuration
    alert_threshold_exceeded = models.BooleanField(default=False)
    requires_manual_review = models.BooleanField(default=False)
    auto_flagged = models.BooleanField(default=False)
    
    # Tracking
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Human verification
    is_verified = models.BooleanField(default=False)
    verified_at = models.DateTimeField(null=True, blank=True)
    verified_by = models.ForeignKey(
        User, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='verified_predictions'
    )
    verification_notes = models.TextField(blank=True)

    class Meta:
        db_table = 'fraud_predictions'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['is_fraudulent']),
            models.Index(fields=['risk_level']),
            models.Index(fields=['model_version']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_verified']),
            # New indexes for enhanced fields
            models.Index(fields=['prediction_method']),
            models.Index(fields=['multi_modal_risk_score']),
            models.Index(fields=['requires_manual_review']),
            models.Index(fields=['alert_threshold_exceeded']),
            models.Index(fields=['ai_generated_probability']),
        ]

    def __str__(self):
        return f"Prediction for Job #{self.job_posting.job_id} - {self.risk_level} Risk"

    @property
    def confidence_percentage(self):
        return f"{self.confidence_score * 100:.1f}%"
    
    @property
    def fraud_percentage(self):
        return f"{self.fraud_probability * 100:.1f}%"
    
    @property
    def multi_modal_percentage(self):
        if self.multi_modal_risk_score is not None:
            return f"{self.multi_modal_risk_score * 100:.1f}%"
        return "N/A"
    
    @property
    def ai_generated_percentage(self):
        if self.ai_generated_probability is not None:
            return f"{self.ai_generated_probability * 100:.1f}%"
        return "N/A"
    
    def mark_verified(self, user=None, notes=''):
        """Mark prediction as human-verified"""
        self.is_verified = True
        self.verified_at = timezone.now()
        if user:
            self.verified_by = user
        if notes:
            self.verification_notes = notes
        self.save()
    
    def get_top_risk_factors(self, limit=3):
        """Get top contributing risk factors"""
        if not self.contributing_factors:
            return []
        
        # Sort by impact level and return top factors
        factors = self.contributing_factors
        if isinstance(factors, list) and factors:
            return factors[:limit]
        return []
    
    @property
    def analysis_completeness(self):
        """Calculate how complete the analysis is (0-1)"""
        components = [
            self.sentiment_score is not None,
            self.logo_verification_score is not None,
            self.vertex_ai_confidence is not None,
            bool(self.company_verification_flags),
            self.ai_generated_probability is not None
        ]
        return sum(components) / len(components)


class UserFeedback(models.Model):
    """Store user corrections and feedback to improve model accuracy"""
    
    FEEDBACK_TYPES = [
        ('CORRECTION', 'Fraud Classification Correction'),
        ('QUALITY', 'Prediction Quality Rating'),
        ('FEATURE', 'Feature Importance Feedback'),
        ('LOGO_QUALITY', 'Logo Analysis Feedback'),  # New
        ('AI_DETECTION', 'AI Content Detection Feedback'),  # New
        ('COMPANY_VERIFICATION', 'Company Verification Feedback'),  # New
        ('GENERAL', 'General Feedback')
    ]
    
    AGREEMENT_LEVELS = [
        ('STRONGLY_AGREE', 'Strongly Agree'),
        ('AGREE', 'Agree'),
        ('NEUTRAL', 'Neutral'),
        ('DISAGREE', 'Disagree'),
        ('STRONGLY_DISAGREE', 'Strongly Disagree')
    ]
    
    prediction = models.ForeignKey(
        FraudPrediction, 
        on_delete=models.CASCADE,
        related_name='user_feedback'
    )
    
    # User correction data
    feedback_type = models.CharField(max_length=25, choices=FEEDBACK_TYPES)
    user_correction = models.BooleanField(
        null=True, 
        blank=True,
        help_text="User's correction: True if fraud, False if legitimate"
    )
    
    # Feedback details
    agreement_level = models.CharField(
        max_length=20,
        choices=AGREEMENT_LEVELS,
        null=True,
        blank=True
    )
    confidence_rating = models.IntegerField(
        choices=[(i, f"{i} Star{'s' if i != 1 else ''}") for i in range(1, 6)],
        null=True,
        blank=True,
        help_text="User confidence in their correction (1-5 stars)"
    )
    
    # Enhanced feedback for multi-modal system
    component_feedback = models.JSONField(
        default=dict,
        blank=True,
        help_text="Feedback on specific analysis components"
    )
    logo_feedback = models.JSONField(
        default=dict,
        blank=True,
        help_text="Specific feedback about logo analysis"
    )
    
    # Detailed feedback
    feedback_reason = models.TextField(
        blank=True,
        help_text="Why the user thinks the prediction is wrong"
    )
    specific_issues = models.JSONField(
        default=list,
        blank=True,
        help_text="Specific issues identified by user"
    )
    
    # User information (anonymous)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField(blank=True)
    session_id = models.CharField(max_length=100, blank=True)
    
    # Optional user account (if logged in)
    user = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='fraud_feedback'
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    is_processed = models.BooleanField(default=False)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Quality control
    is_spam = models.BooleanField(default=False)
    is_helpful = models.BooleanField(null=True, blank=True)
    helpfulness_score = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )

    class Meta:
        db_table = 'user_feedback'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['feedback_type']),
            models.Index(fields=['is_processed']),
            models.Index(fields=['created_at']),
            models.Index(fields=['user_correction']),
            models.Index(fields=['is_helpful']),
        ]

    def __str__(self):
        return f"Feedback for Job #{self.prediction.job_posting.job_id} - {self.feedback_type}"
    
    def mark_processed(self):
        """Mark feedback as processed for model training"""
        self.is_processed = True
        self.processed_at = timezone.now()
        self.save()


class ModelPerformance(models.Model):
    """Track model performance metrics over time"""
    
    MODEL_TYPES = [
        ('fraud_detector', 'Main Fraud Detector'),
        ('logo_analyzer', 'Logo Analysis Model'),
        ('ai_content_detector', 'AI Content Detection'),
        ('company_verifier', 'Company Verification'),
        ('multi_modal', 'Multi-Modal Ensemble'),
        ('vertex_ai', 'Vertex AI Integration')
    ]
    
    model_version = models.CharField(max_length=50)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES, default='fraud_detector')
    
    # Performance metrics
    accuracy = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    precision = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    recall = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    f1_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    auc_score = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Enhanced metrics for multi-modal system
    logo_analysis_accuracy = models.FloatField(null=True, blank=True)
    ai_detection_accuracy = models.FloatField(null=True, blank=True)
    company_verification_accuracy = models.FloatField(null=True, blank=True)
    multi_modal_improvement = models.FloatField(
        null=True,
        blank=True,
        help_text="Improvement over single-modal approaches"
    )
    
    # Dataset information
    total_predictions = models.PositiveIntegerField()
    true_positives = models.PositiveIntegerField()
    true_negatives = models.PositiveIntegerField()
    false_positives = models.PositiveIntegerField()
    false_negatives = models.PositiveIntegerField()
    
    # Training information
    training_data_size = models.PositiveIntegerField(null=True, blank=True)
    training_time = models.FloatField(null=True, blank=True)
    feature_count = models.PositiveIntegerField(null=True, blank=True)
    
    # Deployment tracking
    date_evaluated = models.DateTimeField(auto_now_add=True)
    date_deployed = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=False)
    
    # Performance notes
    notes = models.TextField(blank=True)
    evaluation_dataset = models.CharField(max_length=100, blank=True)
    
    # Cost and resource tracking
    api_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    processing_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    avg_processing_time = models.FloatField(null=True, blank=True)

    class Meta:
        db_table = 'model_performance'
        ordering = ['-date_evaluated']
        indexes = [
            models.Index(fields=['model_version']),
            models.Index(fields=['model_type']),
            models.Index(fields=['is_active']),
            models.Index(fields=['date_evaluated']),
        ]

    def __str__(self):
        return f"Model {self.model_version} ({self.model_type}) - Accuracy: {self.accuracy:.3f}"
    
    @property
    def specificity(self):
        """Calculate specificity (true negative rate)"""
        if (self.true_negatives + self.false_positives) == 0:
            return 0
        return self.true_negatives / (self.true_negatives + self.false_positives)
    
    @property
    def cost_per_prediction(self):
        """Calculate cost per prediction"""
        if self.total_predictions and (self.api_cost or self.processing_cost):
            total_cost = (self.api_cost or 0) + (self.processing_cost or 0)
            return float(total_cost) / self.total_predictions
        return 0


class DataQualityMetrics(models.Model):
    """Track data quality metrics for continuous monitoring"""
    
    METRIC_TYPES = [
        ('COMPLETENESS', 'Data Completeness'),
        ('ACCURACY', 'Data Accuracy'),
        ('CONSISTENCY', 'Data Consistency'),
        ('VALIDITY', 'Data Validity'),
        ('UNIQUENESS', 'Data Uniqueness'),
        ('LOGO_QUALITY', 'Logo Quality Metrics'),  # New
        ('API_RESPONSE', 'API Response Quality'),  # New
    ]
    
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    metric_value = models.FloatField()
    threshold_value = models.FloatField()
    is_passing = models.BooleanField()
    
    # Context
    table_name = models.CharField(max_length=50)
    column_name = models.CharField(max_length=50, blank=True)
    date_checked = models.DateTimeField(auto_now_add=True)
    
    # Details
    record_count = models.PositiveIntegerField()
    issue_count = models.PositiveIntegerField(default=0)
    notes = models.TextField(blank=True)
    
    # Enhanced metrics
    trend_direction = models.CharField(
        max_length=10,
        choices=[('UP', 'Improving'), ('DOWN', 'Declining'), ('STABLE', 'Stable')],
        null=True,
        blank=True
    )
    impact_level = models.CharField(
        max_length=10,
        choices=[('LOW', 'Low'), ('MEDIUM', 'Medium'), ('HIGH', 'High'), ('CRITICAL', 'Critical')],
        default='LOW'
    )

    class Meta:
        db_table = 'data_quality_metrics'
        ordering = ['-date_checked']
        indexes = [
            models.Index(fields=['metric_type', 'date_checked']),
            models.Index(fields=['is_passing']),
            models.Index(fields=['impact_level']),
        ]

    def __str__(self):
        return f"{self.metric_type} for {self.table_name}: {self.metric_value:.3f}"


class PredictionBatch(models.Model):
    """Track batch prediction jobs for bulk processing"""
    
    batch_id = models.UUIDField(default=uuid.uuid4, unique=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Status tracking
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
        ('CANCELLED', 'Cancelled'),
        ('PARTIALLY_COMPLETED', 'Partially Completed'),  # New status
    ]
    
    status = models.CharField(max_length=25, choices=STATUS_CHOICES, default='PENDING')
    
    # Metrics
    total_jobs = models.PositiveIntegerField(default=0)
    processed_jobs = models.PositiveIntegerField(default=0)
    failed_jobs = models.PositiveIntegerField(default=0)
    
    # Enhanced processing options
    PROCESSING_MODES = [
        ('STANDARD', 'Standard Processing'),
        ('ENHANCED', 'Enhanced with Vertex AI'),
        ('LOGO_ONLY', 'Logo Analysis Only'),
        ('FAST', 'Fast Processing (Basic)')
    ]
    
    processing_mode = models.CharField(
        max_length=20,
        choices=PROCESSING_MODES,
        default='STANDARD'
    )
    
    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Configuration
    model_version = models.CharField(max_length=50, default='latest')
    enable_logo_analysis = models.BooleanField(default=False)
    enable_vertex_ai = models.BooleanField(default=False)
    confidence_threshold = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    
    # Results summary
    fraud_detected = models.PositiveIntegerField(default=0)
    avg_confidence = models.FloatField(null=True, blank=True)
    avg_processing_time = models.FloatField(null=True, blank=True)
    
    # Cost tracking
    total_api_calls = models.PositiveIntegerField(default=0)
    estimated_cost = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    
    # Error tracking
    error_message = models.TextField(blank=True)
    failed_job_ids = models.JSONField(default=list, blank=True)

    class Meta:
        db_table = 'prediction_batches'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['processing_mode']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"Batch {self.name} - {self.status}"
    
    @property
    def progress_percentage(self):
        if self.total_jobs == 0:
            return 0
        return (self.processed_jobs / self.total_jobs) * 100
    
    @property
    def success_rate(self):
        if self.processed_jobs == 0:
            return 0
        return ((self.processed_jobs - self.failed_jobs) / self.processed_jobs) * 100
    
    def calculate_estimated_cost(self):
        """Calculate estimated processing cost based on configuration"""
        base_cost = 0.001  # Base cost per job
        
        if self.enable_vertex_ai:
            base_cost += 0.005  # Additional cost for Vertex AI
        if self.enable_logo_analysis:
            base_cost += 0.002  # Additional cost for Vision API
        
        return float(base_cost * self.total_jobs)


# New model for tracking API usage and costs
class APIUsageTracking(models.Model):
    """Track API usage for cost monitoring and optimization"""
    
    API_TYPES = [
        ('VERTEX_AI_NLP', 'Vertex AI Natural Language'),
        ('VISION_API', 'Cloud Vision API'),
        ('TRANSLATION_API', 'Translation API'),
        ('SPEECH_API', 'Speech-to-Text API')
    ]
    
    api_type = models.CharField(max_length=20, choices=API_TYPES)
    job_posting = models.ForeignKey(
        JobPosting,
        on_delete=models.CASCADE,
        related_name='api_usage',
        null=True,
        blank=True
    )
    batch = models.ForeignKey(
        PredictionBatch,
        on_delete=models.CASCADE,
        related_name='api_usage',
        null=True,
        blank=True
    )
    
    # Usage metrics
    requests_made = models.PositiveIntegerField(default=1)
    data_processed = models.PositiveIntegerField(help_text="Data size in bytes")
    processing_time = models.FloatField(help_text="Processing time in seconds")
    
    # Cost tracking
    estimated_cost = models.DecimalField(max_digits=8, decimal_places=4)
    currency = models.CharField(max_length=3, default='USD')
    
    # Response metadata
    response_size = models.PositiveIntegerField(null=True, blank=True)
    success = models.BooleanField(default=True)
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'api_usage_tracking'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['api_type', 'created_at']),
            models.Index(fields=['success']),
        ]
    
    def __str__(self):
        return f"{self.api_type} - ${self.estimated_cost}"


# New model for system alerts and monitoring
class SystemAlert(models.Model):
    """System alerts for monitoring and maintenance"""
    
    ALERT_TYPES = [
        ('PERFORMANCE', 'Performance Issue'),
        ('COST', 'Cost Threshold Exceeded'),
        ('ERROR_RATE', 'High Error Rate'),
        ('DATA_QUALITY', 'Data Quality Issue'),
        ('SECURITY', 'Security Alert'),
        ('MAINTENANCE', 'Maintenance Required')
    ]
    
    SEVERITY_LEVELS = [
        ('LOW', 'Low'),
        ('MEDIUM', 'Medium'), 
        ('HIGH', 'High'),
        ('CRITICAL', 'Critical')
    ]
    
    alert_type = models.CharField(max_length=20, choices=ALERT_TYPES)
    severity = models.CharField(max_length=10, choices=SEVERITY_LEVELS)
    title = models.CharField(max_length=200)
    description = models.TextField()
    
    # Context
    related_model = models.CharField(max_length=50, blank=True)
    related_id = models.PositiveIntegerField(null=True, blank=True)
    metric_value = models.FloatField(null=True, blank=True)
    threshold_value = models.FloatField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    
    # Resolution
    is_resolved = models.BooleanField(default=False)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'system_alerts'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['alert_type', 'severity']),
            models.Index(fields=['is_active', 'is_resolved']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.severity} {self.alert_type}: {self.title}"
    
    def acknowledge(self, user):
        """Acknowledge the alert"""
        self.is_acknowledged = True
        self.acknowledged_by = user
        self.acknowledged_at = timezone.now()
        self.save()
    
    def resolve(self, notes=''):
        """Mark alert as resolved"""
        self.is_resolved = True
        self.resolved_at = timezone.now()
        if notes:
            self.resolution_notes = notes
        self.save()
