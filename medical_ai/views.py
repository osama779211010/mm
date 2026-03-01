from django.utils import timezone
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
    SecretaryProfileSerializer, AppointmentSerializer, ChatMessageSerializer
)
from .models import (
    DiagnosticResult, UserProfile, DoctorProfile, 
    Branch, SecretaryProfile, Appointment, ChatMessage
)
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
                    DoctorProfile.objects.create(user=user)
                elif role == UserProfile.SECRETARY:
                    pass

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
            
            return Response({
                'token': token.key,
                'user': {
                    'id': user.id,
                    'name': user.first_name,
                    'email': user.email,
                    'role': role
                }
            }, status=status.HTTP_200_OK)
        else:
            return Response({'error': 'Invalid Credentials.'}, status=status.HTTP_404_NOT_FOUND)

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

class UserProfileViewSet(viewsets.ModelViewSet):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer

class DoctorProfileViewSet(viewsets.ModelViewSet):
    queryset = DoctorProfile.objects.all()
    serializer_class = DoctorProfileSerializer

class BranchViewSet(viewsets.ModelViewSet):
    queryset = Branch.objects.all()
    serializer_class = BranchSerializer

class SecretaryProfileViewSet(viewsets.ModelViewSet):
    queryset = SecretaryProfile.objects.all()
    serializer_class = SecretaryProfileSerializer

class AppointmentViewSet(viewsets.ModelViewSet):
    queryset = Appointment.objects.all()
    serializer_class = AppointmentSerializer

class ChatMessageViewSet(viewsets.ModelViewSet):
    serializer_class = ChatMessageSerializer

    def get_queryset(self):
        user = self.request.user
        if not user.is_authenticated:
            return ChatMessage.objects.none()
        
        # All messages where current user is sender OR receiver
        queryset = ChatMessage.objects.filter(models.Q(sender=user) | models.Q(receiver=user))
        
        # Optional: filter further by a specific contact
        with_user_id = self.request.query_params.get('with_user')
        if with_user_id:
            queryset = queryset.filter(models.Q(sender_id=with_user_id) | models.Q(receiver_id=with_user_id))
            
        return queryset

    def perform_create(self, serializer):
        serializer.save(sender=self.request.user)

class DiagnosticResultViewSet(viewsets.ModelViewSet):
    queryset = DiagnosticResult.objects.all()
    serializer_class = DiagnosticResultSerializer

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
                    if res.get('class') == 'Pneumonia':
                        advice = "التحليل الرقمي يشير لاحتمالية وجود التهاب رئوي. يرجى مراجعة طبيب صدرية للفحص السريري والأشعة."
                    elif "Issue Detected" in res.get('class', ''):
                        advice = f"تم رصد علامات قد تشير لـ ({res['class']}). يرجى عرض الأشعة على أخصائي."
                    else:
                        advice = "لا تظهر مؤشرات واضحة للالتهاب الرئوي في هذه الأشعة وفقاً للتحليل الأولي."
                elif diag_type == 'SKIN_CANCER':
                    res, conf = ai_service.predict_skin_cancer(image_path)
                    if res.get('class') == 'Normal/Benign':
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
                temp_record.ai_advice = advice
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
        if not message:
            return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        ai_service = get_ai_service()
        try:
            advice = ai_service.get_ai_advice(message)
            return Response({"advice": advice}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
