#!/bin/bash
# Apply database migrations
python manage.py migrate
# Create default admin if not exists
python create_admin.py
# Start Gunicorn
gunicorn mas_project.wsgi --bind 0.0.0.0:$PORT --timeout 120
