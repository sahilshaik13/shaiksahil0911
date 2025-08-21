import os
import fitz  # PyMuPDF

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest
from dotenv import load_dotenv

from google.cloud import storage, aiplatform
from google.oauth2 import service_account
from vertexai.preview.generative_models import GenerativeModel

# Load environment variables from .env
load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Google Cloud setup
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Explicitly load service account credentials
credentials = service_account.Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH)

# Initialize Vertex AI with credentials
aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION, credentials=credentials)


@csrf_exempt
def upload_resume(request):
    if request.method == "POST" and request.FILES.get("resume"):
        resume_file = request.FILES["resume"]
        file_name = resume_file.name

        try:
            # Upload resume to Google Cloud Storage
            storage_client = storage.Client(credentials=credentials)
            bucket = storage_client.bucket(GCP_BUCKET_NAME)
            blob = bucket.blob(file_name)
            blob.upload_from_file(resume_file, content_type=resume_file.content_type)

            # Save locally and extract text using PyMuPDF
            temp_path = f"/tmp/{file_name}"
            blob.download_to_filename(temp_path)

            with fitz.open(temp_path) as doc:
                text = "".join(page.get_text() for page in doc)

            # Generate AI response from Gemini
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
