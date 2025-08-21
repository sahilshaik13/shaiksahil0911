from google.cloud import storage, aiplatform
import os
from django.shortcuts import render


# Load credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set up GCP
aiplatform.init(project="resume-parser-461705", location=os.getenv("GCP_REGION"))
from vertexai.preview.generative_models import GenerativeModel

def upload_resume(request):
    if request.method == "POST" and request.FILES["resume"]:
        resume_file = request.FILES["resume"]
        file_name = resume_file.name

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(file_name)
        blob.upload_from_file(resume_file)

        # Read content (you'll improve this with PDF parsing later)
        text = blob.download_as_text()

        # Analyze with Gemini
        model = GenerativeModel("gemini-pro")
        prompt = f"Extract name, email, phone, skills, education, and experience from this resume:\n\n{text}"
        response = model.generate_content(prompt)
        
        return render(request, "result.html", {"output": response.text})
    return render(request, "upload.html")
