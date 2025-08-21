from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class JobPosting(models.Model):
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
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'job_postings'
        ordering = ['-created_at']

    def __str__(self):
        return f"#{self.job_id} - {self.title}"

class FraudPrediction(models.Model):
    RISK_LEVELS = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High'),
    ]
    
    job_posting = models.OneToOneField(
        JobPosting, 
        on_delete=models.CASCADE,
        related_name='fraud_prediction'
    )
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
        default='Low'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'fraud_predictions'
        ordering = ['-created_at']

    def __str__(self):
        return f"Prediction for Job #{self.job_posting.job_id} - {self.risk_level} Risk"

    @property
    def confidence_percentage(self):
        return f"{self.confidence_score * 100:.1f}%"
    
    @property
    def fraud_percentage(self):
        return f"{self.fraud_probability * 100:.1f}%"
