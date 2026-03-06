import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from google import genai
from google.genai import types
import time
from dotenv import load_dotenv

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

class AIInferenceService:
    def __init__(self):
        # استخدام مسار نسبي للموديلات لضمان العمل على السيرفر
        self.models_root = os.path.join(os.path.dirname(__file__), 'models')
        
        # تحميل الموديلات بشكل مسبق (Pre-loading) لتسريع الاستجابة الأولى
        self._pneumonia_model = None
        self._skin_cancer_interpreter = None
        
        # إعداد Gemini بالـ SDK الجديد
        self._setup_gemini()
        
    def _setup_gemini(self):
        try:
            # تحميل المفتاح من ملف .env لضمان الأمان
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                print("WARNING: GEMINI_API_KEY not found in environment variables.")
                self._client = None
                return

            self._client = genai.Client(api_key=api_key)
            
            # جلب النماذج المتاحة واختيار المتاح منها تلقائياً لتجنب خطأ 404
            self._model_id = "gemini-1.5-flash" # Default fallback
            try:
                available = [m.name.replace('models/', '') for m in self._client.models.list() if 'gemini' in m.name.lower()]
                if available:
                    # تفضيل موديلات flash أو pro المتوفرة
                    flash_models = [m for m in available if 'flash' in m.lower()]
                    pro_models = [m for m in available if 'pro' in m.lower()]
                    
                    if flash_models:
                        self._model_id = flash_models[0]
                    elif pro_models:
                        self._model_id = pro_models[0]
                    else:
                        self._model_id = available[0]
                    print(f"DEBUG SERVICE: Auto-selected Gemini Model -> {self._model_id}")
            except Exception as e:
                print(f"Warning: Could not list models for auto-selection, using default. Error: {e}")
            
            # تعليمات المساعد الشاملة
            self._system_instruction = """
            أنت "MASA" (اختصار لـ Medical AI Smart Assistant)، المساعد الطبي الذكي الشامل لنظام MAS AI HUB.
            
            المهمة:
            1. تمييز الصور الطبية (Vision): حدد بدقة ما إذا كانت الصورة (X-RAY للصدر)، أو (SKIN آفة جلدية)، أو (INVALID غير طبية).
            2. تقديم معلومات طبية دقيقة: أجب بشمولية عن الأعراض، الأدوية، والصحة العامة.
            
            عند تمييز الصور:
            أجب بكلمة واحدة فقط: [X-RAY, SKIN, INVALID].
            
            شخصيتك:
            خبير طبي متعاطف ومهني. استخدم العربية الفصحى المبسطة.
            """
            print("DEBUG SERVICE: New Gemini SDK Client initialized.")
        except Exception as e:
            print(f"Error initializing Gemini SDK: {e}")
            self._client = None

    @property
    def pneumonia_interpreter(self):
        if self._pneumonia_model is None:
            path = os.path.join(self.models_root, 'pneumo', 'pneumonia_model.tflite')
            if os.path.exists(path):
                self._pneumonia_model = tf.lite.Interpreter(model_path=path)
                self._pneumonia_model.allocate_tensors()
        return self._pneumonia_model

    @property
    def skin_cancer_interpreter(self):
        if self._skin_cancer_interpreter is None:
            path = os.path.join(self.models_root, 'Skin-Cancer-Classification-Tflite-Model-master', 'model.tflite')
            if os.path.exists(path):
                self._skin_cancer_interpreter = tf.lite.Interpreter(model_path=path)
                self._skin_cancer_interpreter.allocate_tensors()
        return self._skin_cancer_interpreter

    def _distinguish_with_gemini(self, image_path):
        """
        استخدام الـ SDK الجديد لتمييز نوع الصورة.
        """
        if not self._client:
            return "ERROR"
            
        try:
            img = Image.open(image_path)
            prompt = "Classify this image: X-RAY (if chest x-ray), SKIN (if skin lesion), or INVALID (not medical). Answer with one word only."
            
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    temperature=0.1
                )
            )
            
            category = response.text.strip().upper()
            # التنظيف في حال كانت الإجابة تحتوي على نقاط أو كلمات إضافية
            for word in ["X-RAY", "SKIN", "INVALID"]:
                if word in category:
                    print(f"DEBUG VISION: Gemini classified as -> {word}")
                    return word
            return "INVALID"
        except Exception as e:
            print(f"Vision Client Error: {e}")
            return "ERROR"

    def predict_pneumonia(self, image_path):
        start_time = time.time()
        category = self._distinguish_with_gemini(image_path)
        
        if category != "X-RAY":
            return {"error": "INVALID_IMAGE", "message": "يبدو أن الصورة ليست أشعة سينية للصدر.", "class": "غير صالحة"}, 0.0

        interpreter = self.pneumonia_interpreter
        if not interpreter: return {"error": "Model Error"}, 0.0

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(output_details[0]['index'])
        p_prob = float(preds[0][0])
        
        label = "مصاب (Pneumonia)" if p_prob > 0.5 else "سليم (Normal)"
        display_conf = p_prob if p_prob > 0.5 else 1.0 - p_prob
        
        print(f"DEBUG: Pneumonia Analyzed in {time.time() - start_time:.2f}s")
        return {
            "class": label, 
            "probability": p_prob, 
            "ai_advice": "تم تأكيد نوع الصورة (أشعة سينية). " + 
                         ("هناك مؤشرات لالتهاب رئوي." if p_prob > 0.5 else "الرئة تظهر بشكل سليم.")
        }, display_conf

    def predict_skin_cancer(self, image_path):
        start_time = time.time()
        category = self._distinguish_with_gemini(image_path)
        
        if category != "SKIN":
            return {"error": "INVALID_IMAGE", "message": "الصورة لا تظهر عليها ملامح جلدية واضحة.", "class": "غير صالحة"}, 0.0

        interpreter = self.skin_cancer_interpreter
        if not interpreter: return {"error": "Model Error"}, 0.0

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        input_data = np.expand_dims((img_array / 127.5) - 1.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        labels = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"]
        top_indices = np.argsort(output_data)[-3:][::-1]
        prob = float(output_data[top_indices[0]])
        
        raw_label = labels[top_indices[0]]
        benign_list = ["Melanocytic nevi", "Benign keratosis-like lesions", "Dermatofibroma", "Vascular lesions"]
        label = "سليم (حميد/طبيعي)" if raw_label in benign_list else f"مصاب محتمل ({raw_label})"

        print(f"DEBUG: Skin Analyzed in {time.time() - start_time:.2f}s")
        return {
            "class": label, 
            "raw_class": raw_label, 
            "probability": prob,
            "ai_advice": f"تم التعرف على المنطقة المصابة. النتيجة الأولية: {label}."
        }, prob

    def get_ai_advice(self, message, history=None):
        if not self._client:
            return "عذراً، خدمة الذكاء الاصطناعي معطلة مؤقتاً."
            
        try:
            # تحويل سياق المحادثة للنظام الجديد
            contents = []
            if history:
                for entry in history:
                    role = "user" if entry.get("sender") == "user" else "model"
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=entry.get("text", ""))]))
            
            # إضافة الرسالة الحالية
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))
            
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_instruction,
                    temperature=0.4
                )
            )
            return response.text
        except Exception as e:
            print(f"Chat API Client Error: {e}")
            return "لم أستطع معالجة طلبك حالياً، يرجى المحاولة لاحقاً."
