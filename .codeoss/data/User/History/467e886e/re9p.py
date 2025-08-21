from google.cloud import storage, aiplatform
import os

# Load credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Set up GCP
aiplatform.init(project="your-gcp-project-id", location=os.getenv("GCP_REGION"))
