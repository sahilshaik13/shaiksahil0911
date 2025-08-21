import os
import fitz  # PyMuPDF for PDF
from docx import Document

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel

from dotenv import load_dotenv
load_dotenv()

# Load GCP credentials
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Vertex AI
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        job_title = request.POST.get("job_title", "").strip()
        resume_file = request.FILES["resume"]
        file_name = resume_file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        try:
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            # Save locally for processing
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            # Extract text
            text = ""
            if file_ext == ".pdf":
                with fitz.open(temp_path) as doc:
                    text = "".join(page.get_text() for page in doc)
            elif file_ext == ".docx":
                doc = Document(temp_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_ext == ".doc":
                return HttpResponseBadRequest("DOC format is not supported. Please upload DOCX or PDF.")
            else:
                return HttpResponseBadRequest("Unsupported file format.")

            model = GenerativeModel("gemini-2.5-flash-preview-05-20")

            # Prompt 1: ATS Analysis
            ats_prompt = f"""
You are an AI career coach. Based on the following resume, evaluate it for the job title "{job_title}". Provide:

1. ATS Score (out of 100)
2. Top 5 relevant skills for the job
3. Missing or weak skills
4. Suggestions to improve resume match

Resume:
{text}
"""
            ats_response = model.generate_content(ats_prompt)

            # Prompt 2: Resume Info Extraction
            extract_prompt = f"""
Extract the following details from the resume:
- Full Name
- Email Address
- Phone Number
- Skills
- Education
- Work Experience

Resume:
{text}
"""
            extract_response = model.generate_content(extract_prompt)

            return render(request, "result.html", {
                "job_title": job_title,
                "ats_output": ats_response.text,
                "resume_output": extract_response.text,
            })

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
