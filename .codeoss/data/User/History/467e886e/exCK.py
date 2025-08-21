import os
import fitz  # PyMuPDF for PDF parsing
from docx import Document

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest

from google.cloud import storage, aiplatform
from vertexai.preview.generative_models import GenerativeModel

from dotenv import load_dotenv
import re

load_dotenv()

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")
GCP_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

aiplatform.init(project=GCP_PROJECT_ID, location=GCP_REGION)


def parse_ats_output(text):
    """
    Parses the ATS output text to extract score, matched skills,
    experience relevance, education match, and suggestions.
    Assumes the Gemini model returns a formatted string.
    """
    score = 0
    skills_matched = []
    experience_relevance = []
    education_match = []
    suggestions = []

    # Extract ATS Score (e.g. "ATS Score: 85/100")
    score_match = re.search(r"ATS Score\s*[:\-]?\s*(\d{1,3})", text, re.I)
    if score_match:
        score = int(score_match.group(1))

    # Extract lists using simple regex patterns (adjust if Gemini format differs)
    def extract_list(section_name):
        pattern = re.compile(
            rf"{section_name}:(.*?)(?:\n\n|$)", re.S | re.I)
        match = pattern.search(text)
        if not match:
            return []
        content = match.group(1).strip()
        # Split lines starting with dash or bullet
        lines = re.findall(r"[-*]\s*(.+)", content)
        return [line.strip() for line in lines if line.strip()]

    skills_matched = extract_list("Top 5 relevant skills")
    experience_relevance = extract_list("Experience Relevance")
    education_match = extract_list("Education Match")
    suggestions = extract_list("Suggestions")

    return {
        "score": score,
        "skills_matched": skills_matched,
        "experience_relevance": experience_relevance,
        "education_match": education_match,
        "suggestions": suggestions,
    }


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

            # Extract text from file
            text = ""
            if file_ext == ".pdf":
                with fitz.open(temp_path) as doc:
                    text = "\n".join(page.get_text() for page in doc)
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
3. Experience Relevance
4. Education Match
5. Suggestions to improve resume match

Resume:
{text}
"""
            ats_response = model.generate_content(ats_prompt).text

            # Parse ATS output to structured data
            ats_data = parse_ats_output(ats_response)

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
            extract_response = model.generate_content(extract_prompt).text

            return render(request, "result.html", {
                "job_title": job_title,
                "ats_raw_output": ats_response,
                "resume_output": extract_response,
                **ats_data,
            })

        except Exception as e:
            return HttpResponseBadRequest(f"Error: {str(e)}")

    return render(request, "upload.html")
