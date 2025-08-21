import os
import fitz  # PyMuPDF
import re
import json  # Import JSON for parsing

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
                "Given this resume, do the following:\n"
                "1. Extract the following:\n"
                "- Full Name\n- Email\n- Phone\n- Skills\n- Education\n- Work Experience\n"
                "2. Identify mistakes and suggestions for improvement.\n"
                "3. Provide an ATS score out of 100.\n\n"
                "Return the results in JSON format:\n"
                "{\n"
                "  \"info\": {\n"
                "    \"full_name\": \"\",\n"
                "    \"email\": \"\",\n"
                "    \"phone\": \"\",\n"
                "    \"skills\": {\n"
                "      \"languages\": \"\",\n"
                "      \"frameworks\": \"\",\n"
                "      \"databases\": \"\",\n"
                "      \"cloud\": \"\",\n"
                "      \"ai_tools\": \"\",\n"
                "      \"version_control\": \"\",\n"
                "      \"other_technical\": \"\",\n"
                "      \"fields_of_interest\": \"\"\n"
                "    }\n"
                "  },\n"
                "  \"education\": {\n"
                "    \"bachelor\": \"\",\n"
                "    \"twelfth\": \"\",\n"
                "    \"tenth\": \"\"\n"
                "  },\n"
                "  \"work_experience\": \"\",\n"
                "  \"mistakes\": [],\n"
                "  \"ats_score\": 0\n"
                "}\n"
                f"Resume:\n{text}\n"
            )

            # Generate the response from the AI model
            response = model.generate_content(prompt).text

            # Log the response for debugging
            print("AI Response:", response)  # Log the response

            # Parse the JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                return HttpResponseBadRequest(f"JSON Decode Error: {str(e)} - Response: {response}")

            # Prepare context for rendering
            context = {
                "info": data.get("info", {}),
                "education": data.get("education", {}),
                "work_experience": data.get("work_experience", "Not found."),
                "mistakes": data.get("mistakes", []),
                "ats_score": data.get("ats_score", "Not found.")
            }

            return render(request, "result.html", context)

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
