from django.contrib import admin
from .models import (
    DiagnosticResult, UserProfile, DoctorProfile, Branch, SecretaryProfile,
    Appointment, ChatMessage, Notification, FCMToken, SystemSetting, AIChatMessage,
    AdBanner
)

@admin.register(AdBanner)
class AdBannerAdmin(admin.ModelAdmin):
    list_display = ('title', 'is_active', 'created_at')
    list_filter = ('is_active',)
    search_fields = ('title', 'subtitle')

@admin.register(SystemSetting)
class SystemSettingAdmin(admin.ModelAdmin):
    list_display = ('key', 'updated_at', 'description')
    search_fields = ('key',)

@admin.register(AIChatMessage)
class AIChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'timestamp', 'message_preview')
    list_filter = ('timestamp',)
    search_fields = ('user__username', 'message', 'response')

    def message_preview(self, obj):
        return obj.message[:50] + "..." if len(obj.message) > 50 else obj.message
    message_preview.short_description = 'Message'

admin.site.register(UserProfile)
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
