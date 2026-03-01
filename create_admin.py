import os
import django

# إعداد بيئة دجانغو
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mas_project.settings')
django.setup()

from django.contrib.auth.models import User
from medical_ai.models import UserProfile
from rest_framework.authtoken.models import Token

def create_admin_user():
    username = 'admin'
    email = 'admin@mas.com'
    password = 'admin'

    if not User.objects.filter(username=username).exists():
        print(f"Creating superuser: {username}...")
        user = User.objects.create_superuser(username=username, email=email, password=password)
        user.first_name = "System"
        user.last_name = "Admin"
        user.save()

        # إنشاء ملف تعريف المستخدم (Admin Role)
        UserProfile.objects.get_or_create(user=user, role='DOCTOR') # نعطيه دور دكتور للتحكم الكامل
        
        # إنشاء توكن
        Token.objects.get_or_create(user=user)
        
        print(f"SUCCESS: Admin user created with password: {password}")
    else:
        print(f"INFO: User '{username}' already exists.")

if __name__ == '__main__':
    create_admin_user()
