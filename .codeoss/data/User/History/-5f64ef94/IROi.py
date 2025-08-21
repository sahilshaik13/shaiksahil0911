
from django.core.management.base import BaseCommand
from django.utils import timezone
from app.models import JobPosting, FraudPrediction
import csv

class Command(BaseCommand):
    help = 'Load job postings and predictions from a CSV file'

    def add_arguments(self, parser):
        parser.add_argument('csv_filepath', type=str, help='Path to the CSV file')

    def handle(self, *args, **kwargs):
        csv_filepath = kwargs['csv_filepath']

        with open(csv_filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                job = JobPosting(
                    title=row.get('title', ''),
                    location=row.get('location', ''),
                    department=row.get('department', ''),
                    salary_range=row.get('salary_range', ''),
                    company_profile=row.get('company_profile', ''),
                    description=row.get('description', ''),
                    requirements=row.get('requirements', ''),
                    benefits=row.get('benefits', ''),
                    telecommuting=bool(int(row.get('telecommuting', '0'))),
                    has_company_logo=bool(int(row.get('has_company_logo', '0'))),
                    has_questions=bool(int(row.get('has_questions', '0'))),
                    employment_type=row.get('employment_type', ''),
                    required_experience=row.get('required_experience', ''),
                    required_education=row.get('required_education', ''),
                    industry=row.get('industry', ''),
                    function=row.get('function', ''),
                    created_at=timezone.now()
                )
                job.save()

                fraudulent = bool(int(row.get('fraudulent', '0')))
                prediction = FraudPrediction(
                    job=job,
                    is_fraudulent=fraudulent,
                    confidence_score=1.0 if fraudulent else 0.0,
                    fraud_probability=1.0 if fraudulent else 0.0,
                    risk_level='High' if fraudulent else 'Low',
                    created_at=timezone.now()
                )
                prediction.save()

                count += 1
                if count % 100 == 0:
                    self.stdout.write(f"Loaded {count} records...")

        self.stdout.write(self.style.SUCCESS(f"Successfully loaded {count} records from {csv_filepath}"))
