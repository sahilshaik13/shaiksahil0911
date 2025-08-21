import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest
from dotenv import load_dotenv

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel

import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Set up Google Cloud configuration
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Initialize Vertex AI (once globally)
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)


@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        file_name = resume_file.name

        try:
            # Upload to GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            # Download file from GCS
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            # Extract text from PDF
            doc = fitz.open(temp_path)
            text = ""
            for page in doc:
                text += page.get_text()

            # Initialize Gemini model — NO project/location here
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
