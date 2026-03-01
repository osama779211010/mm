from django.contrib import admin
from .models import (
    DiagnosticResult, UserProfile, DoctorProfile, 
    Branch, SecretaryProfile, Appointment, ChatMessage
)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role')
    list_filter = ('role',)

@admin.register(DoctorProfile)
class DoctorProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'specialty', 'level', 'is_responding')
    search_fields = ('user__username', 'specialty')

@admin.register(Branch)
class BranchAdmin(admin.ModelAdmin):
    list_display = ('doctor', 'governorate', 'street_name')
    list_filter = ('governorate',)

@admin.register(SecretaryProfile)
class SecretaryProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'branch')

@admin.register(Appointment)
class AppointmentAdmin(admin.ModelAdmin):
    list_display = ('patient', 'branch', 'appointment_date', 'status')
    list_filter = ('status', 'appointment_date')

@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('sender', 'receiver', 'timestamp', 'is_read')
    list_filter = ('is_read',)

@admin.register(DiagnosticResult)
class DiagnosticResultAdmin(admin.ModelAdmin):
    list_display = ('diagnosis_type', 'user', 'confidence', 'created_at')
