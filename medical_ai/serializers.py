from django.contrib.auth.models import User
from rest_framework import serializers
from .models import (
    Branch, SecretaryProfile, Appointment, ChatMessage, Notification
)

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name')

class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username', read_only=True)
    user_details = UserSerializer(source='user', read_only=True)

    class Meta:
        model = UserProfile
        fields = '__all__'

class BranchSerializer(serializers.ModelSerializer):
    doctor_name = serializers.CharField(source='doctor.user.username', read_only=True)

    class Meta:
        model = Branch
        fields = '__all__'
        read_only_fields = ('doctor',)

class DoctorProfileSerializer(serializers.ModelSerializer):
    user_details = UserSerializer(source='user', read_only=True)
    branches = BranchSerializer(many=True, read_only=True)

    class Meta:
        model = DoctorProfile
        fields = '__all__'

class SecretaryProfileSerializer(serializers.ModelSerializer):
    user_details = UserSerializer(source='user', read_only=True)
    branch_name = serializers.CharField(source='branch.governorate', read_only=True)
    branch_phone = serializers.CharField(source='branch.contact_number', read_only=True)

    class Meta:
        model = SecretaryProfile
        fields = '__all__'

class AppointmentSerializer(serializers.ModelSerializer):
    patient_name = serializers.CharField(source='patient.username', read_only=True)
    patient_full_name = serializers.CharField(source='patient.get_full_name', read_only=True)
    patient_email = serializers.EmailField(source='patient.email', read_only=True)
    branch_name = serializers.CharField(source='branch.governorate', read_only=True)
    branch_phone = serializers.CharField(source='branch.contact_number', read_only=True)

    class Meta:
        model = Appointment
        fields = '__all__'

class ChatMessageSerializer(serializers.ModelSerializer):
    sender_name = serializers.CharField(source='sender.username', read_only=True)
    receiver_name = serializers.CharField(source='receiver.username', read_only=True)

    class Meta:
        model = ChatMessage
        fields = '__all__'

class DiagnosticResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DiagnosticResult
        fields = '__all__'
        read_only_fields = ('result', 'confidence', 'ai_advice', 'created_at')

class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()
    diagnosis_type = serializers.ChoiceField(choices=DiagnosticResult.DIAGNOSIS_TYPES)

class NotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Notification
        fields = '__all__'
