from django.utils import timezone
from django.db import models
from datetime import timedelta
from django.db.models import Count, Avg
from rest_framework import viewsets, status, permissions
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from .serializers import (
    ImageUploadSerializer, DiagnosticResultSerializer, UserSerializer,
    UserProfileSerializer, DoctorProfileSerializer, BranchSerializer,
    SecretaryProfileSerializer, AppointmentSerializer, ChatMessageSerializer,
    NotificationSerializer, FCMTokenSerializer, AIChatMessageSerializer,
    AdBannerSerializer
)
import requests
import json
from .models import (
    DiagnosticResult, UserProfile, DoctorProfile, 
    Branch, SecretaryProfile, Appointment, ChatMessage, Notification, FCMToken,
    AIChatMessage, AdBanner
)

class AdBannerViewSet(viewsets.ReadOnlyModelViewSet):
    # Public viewset for active ads
    queryset = AdBanner.objects.filter(is_active=True).order_by('-created_at')
    serializer_class = AdBannerSerializer
    permission_classes = [permissions.AllowAny]
from django.contrib.auth.models import User

from django.db import transaction

class RegisterView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        name = request.data.get('name')
        role = request.data.get('role', 'PATIENT')

        if not email or not password or not name:
            return Response({'error': 'الرجاء إدخال جميع البيانات المطلوبة.'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=email).exists():
            return Response({'error': 'هذا البريد الإلكتروني مسجل بالفعل. حاول تسجيل الدخول.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            with transaction.atomic():
                # Create user
                user = User.objects.create_user(username=email, email=email, password=password, first_name=name)
                
                # Create UserProfile
                user_profile = UserProfile.objects.create(user=user, role=role)
                
                # Create extra profile based on role
                if role == UserProfile.DOCTOR:
                    specialty = request.data.get('specialty', '').strip() or 'طبيب عام'
                    bio = request.data.get('bio', '').strip()
                    level = request.data.get('level', 'BACHELOR')
                    DoctorProfile.objects.create(user=user, specialty=specialty, bio=bio, level=level)
                elif role == UserProfile.SECRETARY:
                    branch_id = request.data.get('branch_id')
                    if not branch_id:
                        return Response({'error': 'يجب تحديد الفرع للسكرتير.'}, status=status.HTTP_400_BAD_REQUEST)
                    try:
                        branch = Branch.objects.get(id=branch_id)
                        SecretaryProfile.objects.create(user=user, branch=branch)
                    except Branch.DoesNotExist:
                        return Response({'error': 'الفرع المحدد غير موجود.'}, status=status.HTTP_400_BAD_REQUEST)
                    except Exception as e:
                         return Response({'error': f'خطأ في تعيين الفرع: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

                token, created = Token.objects.get_or_create(user=user)
                
                return Response({
                    'token': token.key,
                    'user': {
                        'id': user.id,
                        'name': user.first_name,
                        'email': user.email,
                        'role': role
                    }
                }, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({'error': f'حدث خطأ أثناء التسجيل: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate(username=email, password=password)
        if user is not None:
            token, created = Token.objects.get_or_create(user=user)
            profile = getattr(user, 'userprofile', None)
            role = profile.role if profile else 'PATIENT'
            
            response_data = {
                'token': token.key,
                'user': {
                    'id': user.id,
                    'name': user.first_name,
                    'email': user.email,
                    'role': role
                }
            }

            # Add branch info for secretaries
            if role == UserProfile.SECRETARY:
                sec_profile = getattr(user, 'secretary_profile', None)
                if sec_profile:
                    response_data['user']['branch_id'] = sec_profile.branch.id
                    response_data['user']['branch_name'] = f"{sec_profile.branch.governorate} - {sec_profile.branch.street_name}"

            return Response(response_data, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Invalid Credentials.'}, status=status.HTTP_401_UNAUTHORIZED)

# تهيئة الخدمة بشكل متأخر (Lazy Loading) لتجنب التحميل عند بدء السيرفر أو عمل Migrations
_ai_service = None

def get_ai_service():
    global _ai_service
    if _ai_service is None:
        from .inference_services import AIInferenceService
        _ai_service = AIInferenceService()
    return _ai_service

class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAdminUser]

class UserProfileViewSet(viewsets.ModelViewSet):
    serializer_class = UserProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            return UserProfile.objects.all()
        return UserProfile.objects.filter(user=self.request.user)

class DoctorProfileViewSet(viewsets.ModelViewSet):
    queryset = DoctorProfile.objects.all()
    serializer_class = DoctorProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

class BranchViewSet(viewsets.ModelViewSet):
    serializer_class = BranchSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        profile = getattr(user, 'userprofile', None)
        
        if user.is_staff:
            return Branch.objects.select_related('doctor__user').all()
        
        if profile and profile.role == UserProfile.DOCTOR:
            return Branch.objects.select_related('doctor__user').filter(doctor__user=user)
        
        if profile and profile.role == UserProfile.SECRETARY:
            sec_profile = getattr(user, 'secretary_profile', None)
            if sec_profile:
                return Branch.objects.select_related('doctor__user').filter(id=sec_profile.branch.id)
        
        # Patients can see branches to book them, but maybe filter by doctor_id in query params
        doctor_id = self.request.query_params.get('doctor_id')
        if doctor_id:
            return Branch.objects.select_related('doctor__user').filter(doctor_id=doctor_id)
            
        return Branch.objects.select_related('doctor__user').all()

    def perform_create(self, serializer):
        # Auto-assign branch to the doctor who created it
        if hasattr(self.request.user, 'doctor_profile'):
            serializer.save(doctor=self.request.user.doctor_profile)
        else:
            serializer.save()

class SecretaryProfileViewSet(viewsets.ModelViewSet):
    serializer_class = SecretaryProfileSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        if user.is_staff:
            return SecretaryProfile.objects.select_related('user', 'branch').all()
        
        # Doctors see secretaries in their branches
        if hasattr(user, 'doctor_profile'):
            return SecretaryProfile.objects.select_related('user', 'branch').filter(branch__doctor__user=user)
            
        # Secretary sees themselves
        if hasattr(user, 'secretary_profile'):
            return SecretaryProfile.objects.select_related('user', 'branch').filter(user=user)
            
        return SecretaryProfile.objects.none()

class AppointmentViewSet(viewsets.ModelViewSet):
    serializer_class = AppointmentSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        user = self.request.user
        profile = getattr(user, 'userprofile', None)

        # Optimization: Fetch patient information and branch details in one go
        base_qs = Appointment.objects.select_related('patient', 'branch__doctor__user')

        if user.is_staff:
            return base_qs.all()

        if profile and profile.role == UserProfile.DOCTOR:
            return base_qs.filter(branch__doctor__user=user)

        if profile and profile.role == UserProfile.SECRETARY:
            sec_profile = getattr(user, 'secretary_profile', None)
            if sec_profile:
                return base_qs.filter(branch=sec_profile.branch)

        # For patients
        return base_qs.filter(patient=user)

    def perform_create(self, serializer):
        appointment = serializer.save(patient=self.request.user)
        # إشعار للسكرتارية في هذا الفرع
        secretaries = User.objects.filter(secretary_profile__branch=appointment.branch)
        for sec in secretaries:
            Notification.objects.create(
                receiver=sec,
                title="طلب حجز جديد",
                message=f"لديك طلب حجز جديد من المريض {self.request.user.username} بتاريخ {appointment.appointment_date.date()}"
            )

    def perform_update(self, serializer):
        old_instance = self.get_object()
        appointment = serializer.save()
        
        if old_instance.status != appointment.status:
            # إشعار للمريض
            status_ar = "مقبول" if appointment.status == 'APPROVED' else "مرفوض" if appointment.status == 'REJECTED' else "مكتمل"
            Notification.objects.create(
                receiver=appointment.patient,
                title="تحديث حالة الحجز",
                message=f"تم تحديث حالة حجزك بتاريخ {appointment.appointment_date.date()} إلى: {status_ar}"
            )
            # --- تفعيل الإشعارات الفورية (Push Notifications) ---
            send_fcm_notification(
                appointment.patient,
                "تحديث حالة الحجز",
                f"تم تحديث حالة حجزك بتاريخ {appointment.appointment_date.date()} إلى: {status_ar}"
            )

def send_fcm_notification(user, title, message):
    """
    وظيفة مساعدة لإرسال إشعارات Firebase
    """
    tokens = FCMToken.objects.filter(user=user).values_list('token', flat=True)
    if not tokens:
        return

    # ملاحظة: يتطلب هذا مفتاح سيرفر Firebase (Server Key) أو ملف الإعدادات
    # سنضع الهيكل الأساسي هنا لاستخدامه مع Firebase Cloud Messaging
    url = "https://fcm.googleapis.com/fcm/send"
    # هذا المفتاح تجريبي للبرمجة، يجب استبداله بمفتاح حقيقي من Firebase Console
    headers = {
        "Content-Type": "application/json",
        "Authorization": "key=YOUR_FIREBASE_SERVER_KEY"
    }

    for token in tokens:
        payload = {
            "to": token,
            "notification": {
                "title": title,
                "body": message,
                "sound": "default"
            },
            "data": {
                "click_action": "FLUTTER_NOTIFICATION_CLICK"
            }
        }
        try:
            requests.post(url, headers=headers, data=json.dumps(payload), timeout=5)
        except Exception as e:
            print(f"FCM Error: {e}")

class FCMTokenViewSet(viewsets.ModelViewSet):
    serializer_class = FCMTokenSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return FCMToken.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # تحديث أو إنشاء التوكن الخاص بالمستخدم وحذف القديم إذا وجد
        token_val = self.request.data.get('token')
        if token_val:
            FCMToken.objects.filter(token=token_val).delete()
        serializer.save(user=self.request.user)

class NotificationViewSet(viewsets.ModelViewSet):
    serializer_class = NotificationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Notification.objects.filter(receiver=self.request.user)

    def perform_create(self, serializer):
        # Prevent users from creating notifications for others via API
        serializer.save(receiver=self.request.user)

class ChatMessageViewSet(viewsets.ModelViewSet):
    serializer_class = ChatMessageSerializer

    def get_queryset(self):
        user = self.request.user
        if not user.is_authenticated:
            return ChatMessage.objects.none()
        
        # All messages where current user is sender OR receiver
        queryset = ChatMessage.objects.select_related('sender', 'receiver') \
            .filter(models.Q(sender=user) | models.Q(receiver=user))
        
        # Optional: filter further by a specific contact
        with_user_id = self.request.query_params.get('with_user')
        if with_user_id:
            queryset = queryset.filter(models.Q(sender_id=with_user_id) | models.Q(receiver_id=with_user_id))
            
        return queryset

    def perform_create(self, serializer):
        serializer.save(sender=self.request.user)

class DiagnosticResultViewSet(viewsets.ModelViewSet):
    serializer_class = DiagnosticResultSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        if self.request.user.is_staff:
            return DiagnosticResult.objects.all()
        return DiagnosticResult.objects.filter(user=self.request.user)

class AdminStatsView(APIView):
    permission_classes = [permissions.IsAdminUser]

    def get(self, request):
        today = timezone.now().date()
        seven_days_ago = today - timedelta(days=7)

        # Basic Stats
        total_users = User.objects.count()
        today_diagnoses = DiagnosticResult.objects.filter(created_at__date=today).count()
        avg_confidence = DiagnosticResult.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0
        total_appointments = Appointment.objects.count()

        # Historical Chart Data (Last 7 days)
        history = DiagnosticResult.objects.filter(created_at__date__gte=seven_days_ago) \
            .extra(select={'day': "date(created_at)"}) \
            .values('day') \
            .annotate(count=Count('id')) \
            .order_by('day')

        # Recent Queue
        recent_results = DiagnosticResult.objects.all().order_by('-created_at')[:5]
        recent_data = []
        for r in recent_results:
            recent_data.append({
                'id': f"#REQ-{r.id}",
                'user': r.user.get_full_name() or r.user.username if r.user else "ضيف",
                'type': r.diagnosis_type,
                'time': r.created_at.strftime("%Y-%m-%d %H:%M"),
                'confidence': f"{r.confidence * 100:.1f}%",
                'status': 'مكتمل'
            })

        return Response({
            'stats': {
                'total_users': total_users,
                'today_diagnoses': today_diagnoses,
                'ai_accuracy': f"{avg_confidence * 100:.1f}%",
                'server_pressure': '28%', # Mocked system load
            },
            'chart_data': list(history),
            'recent_queue': recent_data
        })

class MedicalDiagnosisView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        ai_service = get_ai_service()
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image_file = serializer.validated_data['image']
            diag_type = serializer.validated_data['diagnosis_type']
            print(f"DEBUG VIEW: Incoming request - Type: {diag_type}, Image: {image_file.name}")

            # حفظ سجل مؤقت
            temp_record = DiagnosticResult.objects.create(
                user=request.user if request.user.is_authenticated else None,
                diagnosis_type=diag_type,
                image=image_file,
                result={},
                confidence=0.0,
                ai_advice=""
            )
            image_path = temp_record.image.path

            # --- استدعاء نموذج الذكاء الاصطناعي الحقيقي ---
            try:
                if diag_type == 'PNEUMONIA':
                    res, conf = ai_service.predict_pneumonia(image_path)
                    res_class = res.get('class', '').lower()
                    if 'pneumonia' in res_class or 'مصاب' in res_class:
                        advice = "التحليل الرقمي يشير لاحتمالية وجود التهاب رئوي. يرجى مراجعة طبيب صدرية للفحص السريري والأشعة."
                    elif 'invalid' in res_class or 'unconfirmed' in res_class or 'غير مؤكد' in res_class:
                        advice = res.get('message', "النتيجة غير حاسمة، يرجى رفع صورة أشعة أكثر دقة.")
                    else:
                        advice = "لا تظهر مؤشرات واضحة للالتهاب الرئوي في هذه الأشعة وفقاً للتحليل الأولي."
                elif diag_type == 'SKIN_CANCER':
                    res, conf = ai_service.predict_skin_cancer(image_path)
                    res_class = res.get('class', '').lower()
                    if 'normal' in res_class or 'benign' in res_class or 'غير مصاب' in res_class:
                        advice = "التصبغات الجلدية تبدو حميدة أو طبيعية. ومع ذلك، يفضل مراجعة الطبيب في حال تغير شكلها أو لونها."
                    else:
                        advice = f"تم رصد احتمالية إصابة بـ ({res['class']}). يرجى مراجعة طبيب جلدية متخصص للفحص السريري الدقيق."
                elif diag_type == 'BRAIN_TUMOR':
                    res, conf = ai_service.predict_brain_tumor(image_path)
                    advice = "يظهر التحليل الأولي استقراراً في الأنسجة (وضع تجريبي). يرجى مراجعة طبيب الأعصاب المختص."
                else:
                    return Response({"error": "Unknown diagnosis type"}, status=status.HTTP_400_BAD_REQUEST)

                temp_record.result = res
                temp_record.confidence = conf
                temp_record.ai_advice = res.get('ai_advice', advice)
                temp_record.save()

                return Response(DiagnosticResultSerializer(temp_record).data, status=status.HTTP_201_CREATED)
            except Exception as e:
                import traceback
                traceback.print_exc()
                # في حالة حدوث خطأ في المعالجة، نعيد رسالة خطأ
                return Response({"error": f"AI Processing Error: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ChatAdviceView(APIView):
    def post(self, request, *args, **kwargs):
        message = request.data.get('message', '')
        history = request.data.get('history', []) # استقبال سجل المحادثة
        if not message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        ai_service = get_ai_service()
        try:
            advice = ai_service.get_ai_advice(message, history=history)
            return Response({"advice": advice}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class AIChatViewSet(viewsets.ModelViewSet):
    serializer_class = AIChatMessageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AIChatMessage.objects.filter(user=self.request.user).order_by('timestamp')

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class ChatAdviceView(APIView):
    # This view will now also persist the chat
    def post(self, request):
        message = request.data.get('message')
        history = request.data.get('history', [])
        
        if not message:
            return Response({'error': 'Message is required'}, status=400)

        ai_service = get_ai_service()
        try:
            # Get response from Gemini
            response_text = ai_service.get_chat_advice(message, history)
            
            # Persist to database
            if request.user.is_authenticated:
                AIChatMessage.objects.create(
                    user=request.user,
                    message=message,
                    response=response_text
                )

            return Response({'response': response_text})
        except Exception as e:
            return Response({'error': str(e)}, status=500)
