# Set up virtual environment
python -m venv venv
source venv/bin/activate
# Create Django project
django-admin startproject resume_ai
cd resume_ai
python manage.py startapp parser
pip install django python-dotenv google-cloud-storage google-cloud-aiplatform
# Set up virtual environment
python -m venv venv
source venv/bin/activate
# Create Django project
django-admin startproject resume_ai
cd resume_ai
python manage.py startapp parser
# go to your outer project folder
cd resume_ai
# move the inner resume_ai one level up
mv resume_ai/* .
# delete the now-empty inner folder
rmdir resume_ai
cd resume_ai
mv resume_ai/* .
rmdir resume_ai
cd ~            # Go back to your home directory
rm -rf resume_ai
# Set up virtual environment
python -m venv venv
source venv/bin/activate
# Create Django project
django-admin startproject resume_ai
cd resume_ai
python manage.py startapp parser
GOOGLE_APPLICATION_CREDENTIALS=credentials.json
GCP_BUCKET_NAME=resume-parser-uploads
GCP_REGION=us-central1
import os
from dotenv import load_dotenv
load_dotenv()
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
GCP_REGION = os.getenv("GCP_REGION")
python manage.py migrate
python manage.py runserver
python manage.py migrate
python manage.py runserver
python manage.py migrate
python manage.py runserver
ls -a
source /home/shaiksahil0911/venv/bin/activate
ls -a
cd resume_ai
ls -a
cd parser
ls -a
cd resume_ai
find . -name ".env"
nano .env
ls -a
python manage.py runserver
pip install dotenv
python manage.py runserver
nano .env
python manage.py runserver
nano .env
cd resume_ai
nano .env
pip install PyMuPDF
pip install PyMuPDF
source /home/shaiksahil0911/venv/bin/activate
python manage.py runserver
cd resume_ai
source /home/shaiksahil0911/venv/bin/activate
pip install PyMuPDF
python manage.py runserver
source /home/shaiksahil0911/venv/bin/activate
cd resume_ai
pip freeze > requirements.txt
cd resume_ai
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud artifacts repositories create resume-parser-repo   --repository-format=docker   --location=us-central1
gcloud artifacts repositories create resume-parser-repo   --repository-format=docker   --location=us-central1
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud artifacts repositories create resume-parser-repo   --repository-format=docker   --location=us-central1   --description="Docker repository for Resume Parser"
gcloud auth configure-docker us-central1-docker.pkg.dev
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
pip freeze > requirements.txt
# RUN python manage.py collectstatic --noinput
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings   --port=8080
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings,PORT=8080
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings   --port=8080
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --platform=managed   --region=us-central1   --allow-unauthenticated   --set-env-vars=DJANGO_SETTINGS_MODULE=resume_ai.settings
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud logs read --project=resume-parser-461705 --service=resume-parser-service --limit=50
gcloud run deploy resume-parser-service   --timeout=300
gcloud run deploy resume-parser-service   --image=us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --timeout=300   --region=us-central1   --platform=managed
# entrypoint.sh
#!/bin/sh
exec gunicorn resume_ai.wsgi:application --bind 0.0.0.0:$PORT
gcloud run services delete resume-parser-service --region=us-central1
gcloud artifacts repositories list --location=us-central1
gcloud artifacts docker images list us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo
gcloud artifacts docker images delete us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app@sha256:<digest>
gcloud artifacts repositories delete resume-parser-repo --location=us-central1
docker system prune -a
chmod +x entrypoint.sh
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud artifacts repositories create resume-parser-repo   --repository-format=docker   --location=us-central1   --description="Docker repository for resume parser project"
gcloud artifacts repositories create resume-parser-repo   --repository-format=docker   --location=us-central1   --description="Docker repository for resume parser project"
gcloud artifacts repositories list --location=us-central1
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated
gcloud logs read --project=resume-parser-461705 --limit=50 --freshness=1h
gcloud logs read --project=resume-parser-461705 --limit=50 --freshness=1h
gcloud app logs read --project=resume-parser-461705 --limit=50 --freshness=1h
gcloud logs read "projects/resume-parser-461705/logs/run.googleapis.com%2Fcloud-run-revision"   --limit=50   --project=resume-parser-461705
gcloud logs read "projects/resume-parser-461705/logs/run.googleapis.com%2Fcloud-run-revision"   --limit=50   --severity=ERROR   --project=resume-parser-461705
docker build -t resume-parser-app .
docker run -p 8080:8080 resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
cd resume_ai
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
docker run -p 8080:8080 us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
source /home/shaiksahil0911/venv/bin/activate
cat requirements.txt
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker run -p 8080:8080 us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
docker run -p 8080:8080 us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
source /home/shaiksahil0911/venv/bin/activate
cd resume_ai
nano .env
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker run -p 8080:8080 us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300
source /home/shaiksahil0911/venv/bin/activate
resume_ai
ls
cd resume_ai
ls
source /home/shaiksahil0911/venv/bin/activate
cd resume_ai
ls
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account.json"
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com   --project resume-parser-461705
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com   --project resume-parser-461705
nano .env
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --project resume-parser-461705   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com   --set-env-vars GCP_PROJECT_ID=resume-parser-461705,GCP_REGION=us-central1,GCS_BUCKET_NAME=resume-parser-uploads
pip install python-docx
cd resume_ai
pip freeze < requirements.txt
cd resume_ai
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
cd resume_ai
docker build -t us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app .
docker push us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app
gcloud run deploy resume-parser-service   --image us-central1-docker.pkg.dev/resume-parser-461705/resume-parser-repo/resume-parser-app   --region us-central1   --platform managed   --allow-unauthenticated   --timeout=300   --project resume-parser-461705   --service-account resume-parser-service@resume-parser-461705.iam.gserviceaccount.com   --set-env-vars GCP_PROJECT_ID=resume-parser-461705,GCP_REGION=us-central1,GCS_BUCKET_NAME=resume-parser-uploads
ls
cd style-bot
source /home/shaiksahil0911/venv/bin/activate
pip install flask
python app.py
cd style-bot
pip install flask
gcloud app create --region=asia-south1
gcloud config set project $YOUR_PROJECT_ID
gcloud config set project buyforecast
gcloud app create --region=asia-south1
gcloud app browse
gsutil iam ch serviceAccount:buyforecast@appspot.gserviceaccount.com:roles/storage.admin gs://staging.buyforecast.appspot.com
gsutil iam get gs://staging.buyforecast.appspot.com
pip install django
django-admin startproject pathpilot
cd pathpilot
manage.py startapp app
python manage.py startapp app
python manage.py 
python runserver manage.py
python manage.py runserver
pip freeze > requirements.txt
source /home/shaiksahil0911/venv/bin/activate
cd pathpilot
virtualenv env
env/Scripts/activate
cd pathpilot
virtualenv envv
envv/scripts/activate
source env/bin/activate
source envv/bin/activate
pip install django
pip install --upgrade pip
pip install gunicorn
pip freeze > requirements.txt
gcloud run deploy django-hello-world   --source .   --platform managed   --allow-unauthenticated   --port 8080
gcloud run deploy pathpilot   --source .   --platform managed   --allow-unauthenticated   --port 8080
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
# 5. Set default region
gcloud config set run/region us-central1
gcloud storage buckets create gs://movierate1905 --location=us-central1
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud run services describe movierate1905 --region=us-central1 --format="export" | grep -A 20 env
gcloud run services update movierate1905 --region us-central1 --update-env-vars="
NEXT_PUBLIC_SUPABASE_URL=https://mwcfrunialqiedarxckw.supabase.co,NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im13Y2ZydW5pYWxxaWVkYXJ4Y2t3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTQ0MDI3NDEsImV4cCI6MjA2OTk3ODc0MX0.MXgAXYUGJo9jJV22G-Vl6kWfs_RCZRaLahvuhPFN7xE,OPENROUTER_API_KEY=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"
cd pathpilot
python manage.py
python manage.py runserver
python manage.py makemigration
python manage.py makemigrations
python manage.py migrate
python manage.py makemigrations
source env/bin/activate
source envv/bin/activate
pip install textblob pandas numpy scikit-learn
pip freeze > requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
cd pathpilot
mkdir -p app/management/commands
touch app/management/__init__.py
touch app/management/commands/__init__.py
python manage.py load_dataset --file data/fake_job_postings.csv
# Create Cloud SQL instance
gcloud sql instances create fraud-detection-db     --database-version=POSTGRES_14     --tier=db-f1-micro     --region=us-central1     --storage-type=SSD     --storage-size=20GB
# Create database
gcloud sql databases create frauddb --instance=fraud-detection-db
# Create user
gcloud sql users create django-user     --instance=fraud-detection-db     --password=your-secure-password
cd pathpilot
# Create new migrations
python manage.py makemigrations
# Apply migrations
python manage.py migrate
# Create superuser to access admin
python manage.py createsuperuser
python manage.py makemigrations
python manage.py migrate
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
gcloud sql databases create frauddb --instance=fraud-detection-db
gcloud sql users create django-user --instance=fraud-detection-db --password=your-password
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
cloud_sql_proxy -instances=careerpath-469205:us-central1:fraud-detection-db=tcp:5432
python manage.py makemigrations
python manage.py migrate
# List your Cloud SQL instances
gcloud sql instances list
# Get your instance connection name
gcloud sql instances describe fraud-detection-db --format="value(connectionName)"
# Get your instance IP address
gcloud sql instances describe fraud-detection-db --format="value(ipAddresses[0].ipAddress)"
# Replace with your actual project ID and instance details
cloud_sql_proxy -instances=careerpath-469205:us-central1:fraud-detection-db=tcp:5432
# List existing users
gcloud sql users list --instance=fraud-detection-db
# If django-user doesn't exist, create it:
gcloud sql users create django-user     --instance=fraud-detection-db     --password="k6e)dD./LujeUZcy"
# If it exists but password is wrong, reset it:
gcloud sql users set-password django-user     --instance=fraud-detection-db     --password="k6e)dD./LujeUZcy"
python manage.py migrate
# Kill any existing cloud_sql_proxy processes
pkill cloud_sql_proxy
# Check what's using port 5432
sudo lsof -i :5432
# OR
netstat -tulpn | grep 5432
# If PostgreSQL is running locally, stop it
sudo systemctl stop postgresql
# OR on macOS:
brew services stop postgresql
# Download the latest Cloud SQL Proxy v2
curl -o cloud-sql-proxy https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.15.0/cloud-sql-proxy.linux.amd64
# Make it executable
chmod +x cloud-sql-proxy
# Move to PATH (optional)
sudo mv cloud-sql-proxy /usr/local/bin/
# Verify installation
cloud-sql-proxy --version
# NEW v2 syntax (much simpler!)
cloud-sql-proxy careerpath-469205:us-central1:fraud-detection-db
# Use port 5433 instead of 5432
cloud-sql-proxy --port 5433 careerpath-469205:us-central1:fraud-detection-db
python manage.py migrate
# Test Django connection
python manage.py check --database default
# If that works, run migrations
python manage.py makemigrations
python manage.py migrate
# Create superuser
python manage.py createsuperuser
# Start Django server
python manage.py runserver 8080
# Check if any cloud_sql_proxy processes are running
ps aux | grep cloud_sql_proxy
# Check what's listening on ports 5432 and 5433
ss -tlnp | grep 543
# Start Cloud SQL Proxy on port 5433 with verbose logging
cloud-sql-proxy --port 5433 careerpath-469205:us-central1:fraud-detection-db --verbose
# Start Cloud SQL Proxy on port 5433 with verbose logging
cloud-sql-proxy --port 5433 careerpath-469205:us-central1:fraud-detection-db --verbose
# Start Cloud SQL Proxy on port 5433 with verbose logging
cloud-sql-proxy --port 5433 careerpath-469205:us-central1:fraud-detection-db 
# Check if your Cloud SQL instance is actually running
gcloud sql instances describe fraud-detection-db --format="value(state)"
# If it's not RUNNABLE, start it
gcloud sql instances patch fraud-detection-db --activation-policy=ALWAYS
# Check if Cloud SQL Admin API is enabled
gcloud services list --enabled | grep sqladmin
# Enable it if not enabled
gcloud services enable sqladmin.googleapis.com
# Verify your authentication
gcloud auth list
cloud-sql-proxy --port 5433 careerpath-469205:us-central1:fraud-detection-db 
# Get your external IP
curl ifconfig.me
# Add your IP to authorized networks
gcloud sql instances patch fraud-detection-db     --authorized-networks=$(curl -s ifconfig.me)/32
# Get your Cloud SQL instance's public IP
gcloud sql instances describe fraud-detection-db     --format="value(ipAddresses[0].ipAddress)"
# Add your actual current IP address
gcloud sql instances patch fraud-detection-db     --authorized-networks=34.105.33.14/32
# Wait for the operation to complete (usually 1-2 minutes)
# Get the public IP of your Cloud SQL instance
gcloud sql instances describe fraud-detection-db     --format="value(ipAddresses[0].ipAddress)"
cd pathpilot
python manage.py makemigrations
# Test the direct connection (replace with your Cloud SQL IP)
python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='34.58.198.20',
        port='5432', 
        database='frauddb',
        user='django-user', 
        password='k6e)dD./LujeUZcy',
        sslmode='require'
    )
    print('✅ Direct Cloud SQL connection successful!')
    conn.close()
except Exception as e:
    print(f'❌ Connection failed: {e}')
"
python manage.py migrate
# Get the public IP of your Cloud SQL instance
gcloud sql instances describe fraud-detection-db     --format="value(ipAddresses[0].ipAddress)"
python manage.py runserver
cs pathpilot
python manage.py runserver
cd pathpilot
python manage.py runserver
# Run Django migrations
python manage.py makemigrations
python manage.py migrate
# Create superuser
python manage.py createsuperuser
# Start Django server
python manage.py runserver 8080
# Now Django should connect successfully
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver 8080
python manage.py migrate
curl ifconfig.me
python manage.py createsuperuser
hostname -I
python manage.py migrate
cd pathpilot
python manage.py migrate
# Find your project's VPC networks
gcloud compute networks list
# List all subnets with their CIDR ranges
gcloud compute networks subnets list --format="table(name,region,ipCidrRange)"
# Your current external IP
curl ifconfig.me
# Result: 35.247.47.98
# Add your IP to Cloud SQL authorized networks
gcloud sql instances patch fraud-detection-db     --authorized-networks=35.240.224.115/32
python manage.py migrate
cd pathpilot
python manage.py migrate
# Run the management command to load your CSV data
python manage.py load_csv data/fake_job_postings.csv
# basic load (uses default file path)
python manage.py load_csv
# specify your CSV explicitly
python manage.py load_csv --file data/fake_job_postings.csv
# wipe existing rows first, then load
python manage.py load_csv --file data/fake_job_postings.csv --clear
cd pathpilot
pip freeze > requirements.txt
python manage.py runserver 8080
gcloud run deploy pathpilot --source . --region asia-south1 --allow-unauthenticated --platform managed
source envv/source/activate
cd pathpilot
source envv/bin/activate
cd pathpilot
pip freeze > requirements.txt
gcloud run deploy pathpilot --source . --region asia-south1 --allow-unauthenticated --platform managed
gcloud builds submit --tag $IMAGE .
gcloud run deploy pathpilot --image $IMAGE --region us-central1
gcloud run deploy pathpilot --source . --region asia-south1 --allow-unauthenticated --platform managed
gcloud run deploy pathpilot   --source .   --region asia-south1   --platform managed   --allow-unauthenticated
cd pathpilot
python manage.py runserver
curl ifconfig.me
python manage.py runserver
curl ifconfig.me
gcloud run deploy pathpilot --source . --region asia-south1 --allow-unauthenticated --platform managed
cd pathpilot
gcloud app deploy
cd pathpilot
exit
cd pathpilot
gcloud app deploy
gcloud run deploy pathpilot --source . --region asia-south1 --allow-unauthenticated --platform managed
curl ifconfig.me
gcloud run services update pathpilot   --clear-startup-probe   --clear-readiness-probe   --clear-liveness-probe   --region asia-south1
gcloud run services describe pathpilot --region asia-south1
gcloud run services update pathpilot   --startup-probe-type none   --region asia-south1
gcloud beta run services update pathpilot   --no-startup-probe   --region asia-south1
gcloud run services describe pathpilot --region asia-south1 --format export > service.yaml
gcloud run services replace service.yaml --region asia-south1
gcloud run services describe pathpilot --region asia-south1
pip install google-cloud-aiplatform
python manage.py runserver
export GOOGLE_APPLICATION_CREDENTIALS=/home/shaiksahil0911/careerpath-469205-eb2e7e6a8ebf.json
pip install google-generativeai
cd pathpilot
python manage.py runserver 8081
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
python manage.py runserver 8081
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 8081
pip install python-dotenv
pip install google-generativeai
