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
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userprofile', db_index=True)
    role = models.CharField(max_length=20, choices=ROLES, default=PATIENT)

    def __str__(self):
        return f"{self.user.username} - {self.role}"

class DoctorProfile(models.Model):
    LEVELS = [
        ('BACHELOR', 'بكالريوس'),
        ('MASTER', 'ماجستير'),
        ('DOCTORATE', 'دكتوراه'),
        ('SPECIALIST', 'أخصائي'),
        ('CONSULTANT', 'استشاري'),
    ]
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='doctor_profile', db_index=True)
    specialty = models.CharField(max_length=100, help_text="التخصص الطبي")
    level = models.CharField(max_length=20, choices=LEVELS, default='BACHELOR')
    bio = models.TextField(blank=True, null=True, help_text="نبذة عن الدكتور")
    phone_number = models.CharField(max_length=20, blank=True, null=True, help_text="رقم هاتف الطبيب (اختياري)")
    is_responding = models.BooleanField(default=True, help_text="هل الطبيب متاح للرد؟")

    def __str__(self):
        return f"Dr. {self.user.get_full_name() or self.user.username} - {self.specialty}"

class Branch(models.Model):
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE, related_name='branches', db_index=True)
    governorate = models.CharField(max_length=100, help_text="المحافظة")
    street_name = models.CharField(max_length=255, help_text="الشارع")
    contact_number = models.CharField(max_length=20, blank=True, null=True, help_text="رقم هاتف السكرتارية/الفرع (اختياري)")

    class Meta:
        verbose_name_plural = "Branches"

    def __str__(self):
        return f"{self.doctor.user.username} - {self.governorate} ({self.street_name})"

class SecretaryProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='secretary_profile', db_index=True)
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='secretaries', db_index=True)

    def __str__(self):
        return f"Secretary {self.user.username} at {self.branch}"

class Appointment(models.Model):
    STATUS_CHOICES = [
        ('PENDING', 'قيد الانتظار'),
        ('APPROVED', 'مقبول'),
        ('REJECTED', 'مرفوض'),
        ('COMPLETED', 'مكتمل'),
    ]
    patient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='appointments', db_index=True)
    branch = models.ForeignKey(Branch, on_delete=models.CASCADE, related_name='appointments', db_index=True)
    appointment_date = models.DateTimeField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='PENDING')
    notes = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Appointment: {self.patient.username} at {self.branch} on {self.appointment_date}"

class ChatMessage(models.Model):
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_messages', db_index=True)
    receiver = models.ForeignKey(User, on_delete=models.CASCADE, related_name='received_messages', db_index=True)
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
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

    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, db_index=True)
    diagnosis_type = models.CharField(max_length=50, choices=DIAGNOSIS_TYPES)
    image = models.ImageField(upload_to='medical_uploads/', null=True, blank=True)
    result = models.JSONField(help_text="النتيجة التفصيلية من الذكاء الاصطناعي")
    confidence = models.FloatField(help_text="نسبة دقة التنبؤ")
    ai_advice = models.TextField(help_text="النصيحة الطبية المقترحة", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.diagnosis_type} - {self.created_at}"

class Notification(models.Model):
    receiver = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications', db_index=True)
    title = models.CharField(max_length=255)
    message = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Notification for {self.receiver.username}: {self.title}"
class FCMToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='fcm_tokens', db_index=True)
    token = models.TextField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Token for {self.user.username}"

class SystemSetting(models.Model):
    key = models.CharField(max_length=50, unique=True, help_text="اسم الإعداد (مثال: GEMINI_API_KEY)")
    value = models.TextField(help_text="قيمة الإعداد (المفتاح السري)")
    description = models.CharField(max_length=255, blank=True, help_text="وصف الإعداد")
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.key

class AIChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_conversations', db_index=True)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"AI Chat: {self.user.username} - {self.timestamp}"

class AdBanner(models.Model):
    title = models.CharField(max_length=200, verbose_name="العنوان")
    subtitle = models.CharField(max_length=500, verbose_name="العنوان الفرعي")
    image = models.ImageField(upload_to='banners/', verbose_name="صورة الإعلان")
    link_url = models.URLField(blank=True, null=True, verbose_name="رابط خارجي (اختياري)")
    is_active = models.BooleanField(default=True, verbose_name="نشط")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "إعلان متحرك"
        verbose_name_plural = "الإعلانات المتحركة"

    def __str__(self):
        return self.title
