from google.cloud import storage, aiplatform
import os

# Load credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set up GCP
aiplatform.init(project="resume-parser-461705", location=os.getenv("GCP_REGION"))
