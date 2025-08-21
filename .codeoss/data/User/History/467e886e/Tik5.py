import os
import fitz  # PyMuPDF
import mimetypes

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel
from docx import Document  # For DOCX support

from dotenv import load_dotenv
load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)

@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        file_name = resume_file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        try:
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            # Save locally
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            # Extract text based on file type
            text = ""
            if file_ext == ".pdf":
                with fitz.open(temp_path) as doc:
                    text = "".join(page.get_text() for page in doc)
            elif file_ext in [".docx"]:
                doc = Document(temp_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            elif file_ext == ".doc":
                return HttpResponseBadRequest("DOC format is not supported. Please upload DOCX or PDF.")
            else:
                return HttpResponseBadRequest("Unsupported file format.")

            # Gemini model
            model = GenerativeModel("gemini-2.5-flash-preview-05-20")

            prompt = (
                "Extract the following from this resume:\n"
                "- Full Name\n"
                "- Email Address\n"
                "- Phone Number\n"
                "- Skills\n"
                "- Education\n"
                "- Work Experience\n\n"
                f"Resume Content:\n{text}"
            )

            response = model.generate_content(prompt)

            return render(request, "result.html", {"output": response.text})

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
