from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.contrib import messages
from django.conf import settings
import json
import os
from datetime import datetime, timedelta
from .models import JobPosting, FraudPrediction
from .ml_service import EnhancedFraudDetectionService
from .utils import validate_image_file

# Initialize enhanced ML service
fraud_detector = EnhancedFraudDetectionService()

def home(request):
    """Main fraud detection page with enhanced features"""
    context = {
        'vertex_ai_enabled': hasattr(settings, 'GOOGLE_CLOUD_PROJECT'),
        'supported_image_formats': ['jpg', 'jpeg', 'png', 'gif'],
        'max_file_size': '5MB'
    }
    return render(request, 'fraud_detection/home.html', context)

@csrf_exempt
def enhanced_predict_fraud(request):
    """Enhanced fraud prediction with Vertex AI and logo analysis"""
    if request.method == 'POST':
        try:
            # Handle different content types
            if request.content_type and 'application/json' in request.content_type:
                data = json.loads(request.body)
                logo_file = None
            else:
                data = request.POST.dict()
                logo_file = request.FILES.get('company_logo')
                
                # Convert checkbox values
                data['telecommuting'] = 'telecommuting' in request.POST
                data['has_company_logo'] = bool(logo_file)
                data['has_questions'] = 'has_questions' in request.POST

            # Validate logo file if provided
            if logo_file:
                validation_result = validate_image_file(logo_file)
                if not validation_result['valid']:
                    if request.content_type and 'application/json' in request.content_type:
                        return JsonResponse({
                            'success': False, 
                            'error': validation_result['error']
                        })
                    else:
                        messages.error(request, validation_result['error'])
                        return render(request, 'fraud_detection/predict.html')

            # Create job posting
            job_posting = JobPosting.objects.create(
                title=data.get('title', ''),
                location=data.get('location', ''),
                department=data.get('department', ''),
                salary_range=data.get('salary_range', ''),
                company_profile=data.get('company_profile', ''),
                description=data.get('description', ''),
                requirements=data.get('requirements', ''),
                benefits=data.get('benefits', ''),
                telecommuting=data.get('telecommuting', False),
                has_company_logo=data.get('has_company_logo', False),
                has_questions=data.get('has_questions', False),
                employment_type=data.get('employment_type', ''),
                required_experience=data.get('required_experience', ''),
                required_education=data.get('required_education', ''),
                industry=data.get('industry', ''),
                function=data.get('function', ''),
                poster_ip=request.META.get('REMOTE_ADDR', ''),
                user_agent=request.META.get('HTTP_USER_AGENT', '')
            )

            # Save logo if provided
            if logo_file:
                job_posting.company_logo = logo_file
                job_posting.save()

            # Enhanced analysis pipeline
            analysis_results = {}
            
            # 1. Logo analysis
            if logo_file:
                print("Starting logo analysis...")
                logo_analysis = fraud_detector.analyze_company_logo(logo_file)
                job_posting.logo_analysis_result = logo_analysis
                analysis_results['logo_analysis'] = logo_analysis
            else:
                analysis_results['logo_analysis'] = {'has_logo': False}

            # 2. Vertex AI text analysis
            print("Starting Vertex AI text analysis...")
            vertex_analysis = fraud_detector.vertex_ai_text_analysis(data)
            job_posting.vertex_ai_analysis = vertex_analysis
            analysis_results['vertex_analysis'] = vertex_analysis

            # 3. Company verification
            print("Starting company verification...")
            company_verification = fraud_detector.enhanced_company_verification(
                data, analysis_results['logo_analysis']
            )
            analysis_results['company_verification'] = company_verification

            # 4. Multi-modal fraud prediction
            print("Starting multi-modal prediction...")
            prediction = fraud_detector.multi_modal_fraud_prediction(
                data, 
                analysis_results['logo_analysis'],
                analysis_results['vertex_analysis'],
                analysis_results['company_verification']
            )

            # Save updated job posting
            job_posting.save()

            # Save enhanced fraud prediction
            fraud_prediction = FraudPrediction.objects.create(
                job_posting=job_posting,
                is_fraudulent=prediction['is_fraudulent'],
                confidence_score=prediction['confidence'],
                fraud_probability=prediction['fraud_probability'],
                sentiment_score=prediction.get('sentiment_score', 0.0),
                risk_level=prediction['risk_level'],
                logo_verification_score=company_verification.get('verification_score', 0.0),
                vertex_ai_confidence=vertex_analysis.get('confidence', 0.0),
                company_verification_flags=company_verification.get('flags', []),
                multi_modal_risk_score=prediction.get('multi_modal_score', 0.0),
                ai_generated_probability=vertex_analysis.get('ai_generated_probability', 0.0)
            )

            # Format response
            formatted_prediction = {
                'is_fraudulent': prediction['is_fraudulent'],
                'confidence': f"{prediction['confidence']:.1%}",
                'fraud_probability': f"{prediction['fraud_probability']:.1%}",
                'sentiment_score': f"{prediction.get('sentiment_score', 0.0):.2f}",
                'risk_level': prediction['risk_level'],
                'multi_modal_score': f"{prediction.get('multi_modal_score', 0.0):.1%}",
                'ai_generated_probability': f"{vertex_analysis.get('ai_generated_probability', 0.0):.1%}",
                'logo_quality_score': f"{analysis_results['logo_analysis'].get('professional_quality', 0.0):.1%}",
                'company_verification_score': f"{company_verification.get('verification_score', 0.0):.1%}"
            }

            if request.content_type and 'application/json' in request.content_type:
                return JsonResponse({
                    'success': True,
                    'prediction': formatted_prediction,
                    'job_id': job_posting.job_id,
                    'analysis_results': {
                        'vertex_ai_flags': vertex_analysis.get('flags', []),
                        'company_flags': company_verification.get('flags', []),
                        'logo_analysis': analysis_results['logo_analysis']
                    }
                })
            else:
                return render(request, 'fraud_detection/enhanced_result.html', {
                    'prediction': formatted_prediction,
                    'raw_prediction': prediction,
                    'job_posting': job_posting,
                    'fraud_prediction': fraud_prediction,
                    'analysis_results': analysis_results
                })

        except Exception as e:
            print(f"Error in enhanced_predict_fraud: {str(e)}")
            if request.content_type and 'application/json' in request.content_type:
                return JsonResponse({'success': False, 'error': str(e)})
            else:
                messages.error(request, f"Analysis failed: {str(e)}")
                return render(request, 'fraud_detection/error.html', {'error': str(e)})

    return render(request, 'fraud_detection/enhanced_predict.html')

def analytics_dashboard(request):
    """Enhanced analytics dashboard with Vertex AI insights"""
    # Basic statistics
    total_predictions = FraudPrediction.objects.count()
    fraud_count = FraudPrediction.objects.filter(is_fraudulent=True).count()
    
    # Enhanced statistics
    ai_generated_count = FraudPrediction.objects.filter(
        ai_generated_probability__gt=0.7
    ).count()
    
    logo_analyzed_count = FraudPrediction.objects.filter(
        logo_verification_score__isnull=False
    ).count()

    # Risk level distribution
    risk_distribution = FraudPrediction.objects.values('risk_level').annotate(
        count=Count('risk_level')
    ).order_by('risk_level')

    # Multi-modal accuracy insights
    multi_modal_stats = FraudPrediction.objects.aggregate(
        avg_confidence=Count('confidence_score'),
        avg_logo_score=Count('logo_verification_score'),
        avg_ai_probability=Count('ai_generated_probability')
    )

    # Recent predictions with enhanced data
    recent_predictions = FraudPrediction.objects.select_related('job_posting').order_by('-created_at')
    paginator = Paginator(recent_predictions, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Daily fraud detection over last 7 days
    last_week = datetime.now() - timedelta(days=7)
    daily_stats = []
    for i in range(7):
        day = last_week + timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        day_predictions = FraudPrediction.objects.filter(
            created_at__gte=day_start,
            created_at__lt=day_end
        )
        daily_stats.append({
            'date': day.strftime('%Y-%m-%d'),
            'total': day_predictions.count(),
            'fraud': day_predictions.filter(is_fraudulent=True).count(),
            'ai_generated': day_predictions.filter(ai_generated_probability__gt=0.7).count()
        })

    # Industry analysis with AI insights
    industry_stats = JobPosting.objects.exclude(industry='').values('industry').annotate(
        total_jobs=Count('job_id'),
        fraud_count=Count('fraud_prediction', filter=Q(fraud_prediction__is_fraudulent=True)),
        ai_generated_count=Count('fraud_prediction', 
                               filter=Q(fraud_prediction__ai_generated_probability__gt=0.7))
    ).order_by('-total_jobs')[:10]

    # Logo analysis insights
    logo_stats = {
        'total_with_logos': JobPosting.objects.filter(company_logo__isnull=False).count(),
        'professional_logos': FraudPrediction.objects.filter(logo_verification_score__gt=0.7).count(),
        'suspicious_logos': FraudPrediction.objects.filter(logo_verification_score__lt=0.3).count()
    }

    context = {
        'total_predictions': total_predictions,
        'fraud_count': fraud_count,
        'legitimate_count': total_predictions - fraud_count,
        'fraud_percentage': (fraud_count / total_predictions * 100) if total_predictions > 0 else 0,
        'ai_generated_count': ai_generated_count,
        'logo_analyzed_count': logo_analyzed_count,
        'risk_distribution': {item['risk_level']: item['count'] for item in risk_distribution},
        'recent_predictions': page_obj,
        'daily_stats': daily_stats,
        'industry_stats': industry_stats,
        'logo_stats': logo_stats,
        'multi_modal_stats': multi_modal_stats
    }

    return render(request, 'fraud_detection/enhanced_dashboard.html', context)

def job_detail(request, job_id):
    """Enhanced job detail view with full analysis breakdown"""
    try:
        job_posting = JobPosting.objects.get(job_id=job_id)
        fraud_prediction = FraudPrediction.objects.get(job_posting=job_posting)
        
        # Parse analysis results
        logo_analysis = job_posting.logo_analysis_result or {}
        vertex_analysis = job_posting.vertex_ai_analysis or {}
        
        context = {
            'job_posting': job_posting,
            'fraud_prediction': fraud_prediction,
            'logo_analysis': logo_analysis,
            'vertex_analysis': vertex_analysis,
            'verification_flags': fraud_prediction.company_verification_flags
        }
        return render(request, 'fraud_detection/enhanced_job_detail.html', context)
    except (JobPosting.DoesNotExist, FraudPrediction.DoesNotExist):
        return render(request, 'fraud_detection/error.html', {'error': 'Job not found'})

def api_predict_fraud(request):
    """API endpoint for external integrations"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    try:
        data = json.loads(request.body)
        
        # Quick validation
        required_fields = ['title', 'description', 'company_profile']
        for field in required_fields:
            if not data.get(field):
                return JsonResponse({'error': f'Missing required field: {field}'}, status=400)
        
        # Run analysis without saving to database
        vertex_analysis = fraud_detector.vertex_ai_text_analysis(data)
        company_verification = fraud_detector.enhanced_company_verification(data, {})
        
        prediction = fraud_detector.multi_modal_fraud_prediction(
            data, {}, vertex_analysis, company_verification
        )
        
        return JsonResponse({
            'success': True,
            'is_fraudulent': prediction['is_fraudulent'],
            'confidence': prediction['confidence'],
            'fraud_probability': prediction['fraud_probability'],
            'risk_level': prediction['risk_level'],
            'ai_generated_probability': vertex_analysis.get('ai_generated_probability', 0.0),
            'flags': vertex_analysis.get('flags', []) + company_verification.get('flags', [])
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def bulk_analysis(request):
    """Bulk analysis endpoint for processing multiple job postings"""
    if request.method == 'POST':
        try:
            file = request.FILES.get('bulk_file')
            if not file:
                messages.error(request, 'Please upload a CSV file')
                return render(request, 'fraud_detection/bulk_upload.html')
            
            # Process bulk file (CSV expected)
            results = fraud_detector.process_bulk_analysis(file)
            
            return render(request, 'fraud_detection/bulk_results.html', {
                'results': results,
                'total_processed': len(results),
                'fraud_detected': sum(1 for r in results if r['is_fraudulent'])
            })
            
        except Exception as e:
            messages.error(request, f'Bulk analysis failed: {str(e)}')
            return render(request, 'fraud_detection/bulk_upload.html')
    
    return render(request, 'fraud_detection/bulk_upload.html')

# Legacy support - redirect old endpoint to new enhanced version
def predict_fraud(request):
    """Legacy endpoint - redirects to enhanced version"""
    return enhanced_predict_fraud(request)
