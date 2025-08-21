import os
import fitz  # PyMuPDF
import re

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest
from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Vertex AI with default credentials (no explicit creds)
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        file_name = resume_file.name

        try:
            # Upload the resume to Google Cloud Storage
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            # Download the resume to a temporary path
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            # Extract text from the resume
            with fitz.open(temp_path) as doc:
                text = "".join(page.get_text() for page in doc)

            # Prepare the prompt for the AI model
            model = GenerativeModel("gemini-2.5-flash-preview-05-20")
            prompt = (
                "You are a resume screening expert.\n\n"
                "The current year is 2025"
                "Given this resume, do the following:\n"
                "1. Extract the following:\n"
                "- Full Name\n- Email\n- Phone\n- Skills\n- Education\n- Work Experience\n"
                "2. Identify mistakes and suggestions for improvement.\n"
                "3. Provide an ATS score out of 100.\n\n"
                f"Resume:\n{text}\n\n"
                "Respond in sections titled:\n"
                "## Extracted Info\n## Resume Mistakes\n## ATS Score"
            )

            # Generate the response from the AI model
            response = model.generate_content(prompt).text

            # Extract sections using regex
            extracted_info = re.search(r"## Extracted Info\s*(.*?)\s*##", response, re.DOTALL)
            mistakes = re.search(r"## Resume Mistakes\s*(.*?)\s*##", response, re.DOTALL)
            ats_score = re.search(r"## ATS Score\s*(.*)", response, re.DOTALL)

            # Prepare context for rendering
            context = {
                "info": {
                    "full_name": "Extracted full name here",  # Extract full name from the response
                    "email": "Extracted email here",  # Extract email from the response
                    "phone": "Extracted phone here",  # Extract phone from the response
                    "skills": {
                        "languages": "Extracted languages here",  # Extract languages from the response
                        "frameworks": "Extracted frameworks here",  # Extract frameworks from the response
                        "databases": "Extracted databases here",  # Extract databases from the response
                        "cloud": "Extracted cloud services here",  # Extract cloud services from the response
                        "ai_tools": "Extracted AI tools here",  # Extract AI tools from the response
                        "version_control": "Extracted version control here",  # Extract version control from the response
                        "other_technical": "Extracted other technical skills here",  # Extract other technical skills from the response
                        "fields_of_interest": "Extracted fields of interest here"  # Extract fields of interest from the response
                    }
                },
                "education": {
                    "bachelor": "Extracted bachelor info here",  # Extract bachelor info from the response
                    "twelfth": "Extracted 12th info here",  # Extract 12th info from the response
                    "tenth": "Extracted 10th info here"  # Extract 10th info from the response
                },
                "work_experience": "Extracted work experience here",  # Extract work experience from the response
                "mistakes": mistakes.group(1).strip().split('\n') if mistakes else ["Not found."],
                "ats_score": ats_score.group(1).strip() if ats_score else "Not found."
            }

            return render(request, "result.html", context)

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
