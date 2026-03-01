from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    PATIENT = 'PATIENT'
    DOCTOR = 'DOCTOR'
    SECRETARY = 'SECRETARY'
    
    ROLES = [
        (PATIENT, 'مريض'),
        (DOCTOR, 'طبيب'),
        (SECRETARY, 'سكرتير'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userprofile')
    role = models.CharField(max_length=20, choices=ROLES, default=PATIENT)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

class DoctorProfile(models.Model):
    LEVELS = [
        ('CONSULTANT', 'استشاري'),
        ('SPECIALIST', 'أخصائي'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile')
    specialty = models.CharField(max_length=100, help_text="التخصص الطبي")
    level = models.CharField(max_length=20, choices=LEVELS, default='SPECIALIST')
    bio = models.TextField(blank=True, null=True, help_text="نبذة عن الدكتور")
    phone_number = models.CharField(max_length=20, blank=True, null=True, help_text="رقم هاتف الطبيب (اختياري)")
    is_responding = models.BooleanField(default=True, help_text="هل الطبيب متاح للرد؟")

    def __str__(self):
        return f"Dr. {self.user.get_full_name() or self.user.username} - {self.specialty}"

class Branch(models.Model):
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE, related_name='branches')
    governorate = models.CharField(max_length=100, help_text="المحافظة")
    street_name = models.CharField(max_length=255, help_text="الشارع")
    contact_number = models.CharField(max_length=20, blank=True, null=True, help_text="رقم هاتف السكرتارية/الفرع (اختياري)")

    class Meta:
        verbose_name_plural = "Branches"

    def __str__(self):
        return f"{self.doctor.user.username} - {self.governorate} ({self.street_name})"

class SecretaryProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='secretary_profile')
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='secretaries')

    def __str__(self):
        return f"Secretary {self.user.username} at {self.branch}"

class Appointment(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'قيد الانتظار'),
        ('APPROVED', 'مقبول'),
        ('REJECTED', 'مرفوض'),
        ('COMPLETED', 'مكتمل'),
    ]
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='appointments')
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='appointments')
    appointment_date = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Appointment: {self.patient.username} at {self.branch} on {self.appointment_date}"

class ChatMessage(models.Model):
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_messages')
    receiver = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_messages')
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"From {self.sender.username} to {self.receiver.username} at {self.timestamp}"

class DiagnosticResult(models.Model):
    DIAGNOSIS_TYPES = [
        ('PNEUMONIA', 'التهاب رئوي'),
        ('BREAST_CANCER', 'سرطان الثدي'),
        ('ECG', 'تخطيط القلب'),
        ('SKIN_CANCER', 'أمراض الجلد'),
        ('BONE_FRACTURE', 'كسور العظام'),
        ('BRAIN_TUMOR', 'أورام الدماغ'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    diagnosis_type = models.CharField(max_length=50, choices=DIAGNOSIS_TYPES)
    image = models.ImageField(upload_to='medical_uploads/', null=True, blank=True)
    result = models.JSONField(help_text="النتيجة التفصيلية من الذكاء الاصطناعي")
    confidence = models.FloatField(help_text="نسبة دقة التنبؤ")
    ai_advice = models.TextField(help_text="النصيحة الطبية المقترحة", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.diagnosis_type} - {self.created_at}"
