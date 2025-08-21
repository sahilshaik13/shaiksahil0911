#!/bin/sh
exec gunicorn resume_ai.wsgi:application --bind 0.0.0.0:${PORT:-8080}
