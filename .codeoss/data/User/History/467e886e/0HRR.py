import os
import fitz  # PyMuPDF

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Vertex AI with default credentials (no explicit creds)
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

import re

@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        file_name = resume_file.name

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            with fitz.open(temp_path) as doc:
                text = "".join(page.get_text() for page in doc)

            model = GenerativeModel("gemini-2.5-flash-preview-05-20")

            prompt = (
                "You are a resume screening expert.\n\n"
                "Given this resume, do the following:\n"
                "1. Extract the following:\n"
                "- Full Name\n- Email\n- Phone\n- Skills\n- Education\n- Work Experience\n"
                "2. Identify mistakes and suggestions for improvement.\n"
                "3. Provide an ATS score out of 100.\n\n"
                f"Resume:\n{text}\n\n"
                "Respond in sections titled:\n"
                "## Extracted Info\n## Resume Mistakes\n## ATS Score"
            )

            response = model.generate_content(prompt).text

            # Extract sections using regex
            extracted_info = re.search(r"## Extracted Info\s*(.*?)\s*##", response, re.DOTALL)
            mistakes = re.search(r"## Resume Mistakes\s*(.*?)\s*##", response, re.DOTALL)
            ats_score = re.search(r"## ATS Score\s*(.*)", response, re.DOTALL)

            context = {
                "info": extracted_info.group(1).strip() if extracted_info else "Not found.",
                "mistakes": mistakes.group(1).strip() if mistakes else "Not found.",
                "ats_score": ats_score.group(1).strip() if ats_score else "Not found.",
            }

            return render(request, "result.html", context)

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
