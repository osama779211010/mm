from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    MedicalDiagnosisView, ChatAdviceView, AdminStatsView,
    UserProfileViewSet, DoctorProfileViewSet, BranchViewSet,
    SecretaryProfileViewSet, AppointmentViewSet, ChatMessageViewSet,
    UserViewSet, DiagnosticResultViewSet, RegisterView, LoginView,
    NotificationViewSet, FCMTokenViewSet
)

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'profiles', UserProfileViewSet, basename='userprofile')
router.register(r'doctors', DoctorProfileViewSet)
router.register(r'branches', BranchViewSet, basename='branch')
router.register(r'secretaries', SecretaryProfileViewSet, basename='secretaryprofile')
router.register(r'appointments', AppointmentViewSet, basename='appointment')
router.register(r'messages', ChatMessageViewSet, basename='chatmessage')
router.register(r'results', DiagnosticResultViewSet, basename='diagnosticresult')
router.register(r'notifications', NotificationViewSet, basename='notification')
router.register(r'fcm-tokens', FCMTokenViewSet, basename='fcmtoken')

urlpatterns = [
    path('login/', LoginView.as_view(), name='login'),
    path('register/', RegisterView.as_view(), name='register'),
    path('predict/', MedicalDiagnosisView.as_view(), name='medical-predict'),
    path('chat-advice/', ChatAdviceView.as_view(), name='chat-advice'),
    path('stats/', AdminStatsView.as_view(), name='admin-stats'),
    path('', include(router.urls)),
]
