import os
import fitz  # PyMuPDF for PDF
import mimetypes

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel
from docx import Document

from dotenv import load_dotenv
load_dotenv()

# Load GCP config from .env
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

            # Save to local temporary file
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            # Extract text from resume
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
                return HttpResponseBadRequest("Unsupported file format. Only PDF or DOCX allowed.")

            # Gemini Prompt
            prompt = f"""
You are an expert AI resume screener. Analyze the resume for the job title "{job_title}".
Return the following:
1. ATS Score (out of 100)
2. Top 5 skills that match the job
3. Important missing skills
4. Summary feedback to improve the resume

Resume Content:
{text}
"""

            model = GenerativeModel("gemini-2.5-flash-preview-05-20")
            response = model.generate_content(prompt)

            return render(request, "result.html", {
                "job_title": job_title,
                "output": response.text
            })

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
