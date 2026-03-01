#!/bin/bash
# Apply database migrations
python manage.py migrate
# Start Gunicorn
gunicorn mas_project.wsgi --bind 0.0.0.0:$PORT
