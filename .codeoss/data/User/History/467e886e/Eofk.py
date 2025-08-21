import os
import tempfile
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from google.cloud import aiplatform
from django.conf import settings
import fitz  # PDF parser

def upload_resume(request):
    if request.method == 'POST':
        job_title = request.POST.get('job_title')
        uploaded_file = request.FILES['resume']
        
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)

        with open(file_path, 'rb') as f:
            text = extract_text(f, uploaded_file.name)

        # Prompt 1: ATS Evaluation
        ats_prompt = f"""You are an expert ATS system. Analyze the following resume for the job title '{job_title}'. Provide:
1. ATS Score (out of 100)
2. Top 5 matched skills
3. Missing but important skills
4. Summary feedback

Resume:
{text}
"""
        ats_response = get_gemini_response(ats_prompt)

        # Prompt 2: Resume Info Extraction
        analysis_prompt = f"""Extract the following details from the resume:
- Full Name
- Email Address
- Phone Number
- Skills
- Education
- Work Experience

Resume:
{text}
"""
        analysis_response = get_gemini_response(analysis_prompt)

        # Cleanup
        os.remove(file_path)

        return render(request, 'result.html', {
            'job_title': job_title,
            'ats_output': ats_response,
            'resume_output': analysis_response
        })

    return render(request, 'upload.html')


def extract_text(file_obj, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.pdf':
        doc = fitz.open(stream=file_obj.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc])
    elif ext in ['.doc', '.docx']:
        from docx import Document
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_obj.read())
            tmp_path = tmp.name
        doc = Document(tmp_path)
        os.remove(tmp_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    else:
        return ''


def get_gemini_response(prompt):
    from vertexai.preview.generative_models import GenerativeModel
    aiplatform.init(project="your-gcp-project-id", location="us-central1")  # Replace with your actual values
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    response = model.generate_content(prompt)
    return response.text
