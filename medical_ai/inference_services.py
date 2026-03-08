import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from google import genai
from google.genai import types
import time
from dotenv import load_dotenv
import base64
from openai import OpenAI

# تحميل متغيرات البيئة من ملف .env
load_dotenv()

class AIInferenceService:
    def __init__(self):
        # استخدام مسار نسبي للموديلات لضمان العمل على السيرفر
        self.models_root = os.path.join(os.path.dirname(__file__), 'models')
        
        # تحميل الموديلات بشكل مسبق (Pre-loading) لتسريع الاستجابة الأولى
        self._pneumonia_model = None
        self._skin_cancer_interpreter = None
        
        # اعداد عملاء الذكاء الاصطناعي (OpenAI او Gemini)
        self.ai_provider = None
        self._setup_ai_clients()
        
    def _setup_ai_clients(self):
        try:
            # تحميل المفتاح من قاعدة البيانات لضمان الأمان المطلق وحمايته من التسريب عبر GitHub
            from .models import SystemSetting
            
            try:
                setting = SystemSetting.objects.get(key="OPENAI_API_KEY")
                openai_key = setting.value
            except SystemSetting.DoesNotExist:
                openai_key = os.getenv("OPENAI_API_KEY")

            try:
                setting2 = SystemSetting.objects.get(key="GEMINI_API_KEY")
                gemini_key = setting2.value
            except SystemSetting.DoesNotExist:
                gemini_key = os.getenv("GEMINI_API_KEY")

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

            # محاولة تهيئة OpenAI أولاً لأنه أقوى (حسب طلب المستخدم)
            if openai_key and openai_key.strip():
                self._openai_client = OpenAI(api_key=openai_key.strip())
                self.ai_provider = "openai"
                print("DEBUG SERVICE: OpenAI Client enabled successfully.")
                return

            # في حال عدم وجود OpenAI، نعود لاحتياطي Gemini
            if gemini_key and gemini_key.strip():
                self._client = genai.Client(api_key=gemini_key.strip())
                self.ai_provider = "gemini"
                
                # جلب النماذج المتاحة واختيار المتاح منها تلقائياً لتجنب خطأ 404
                self._model_id = "gemini-1.5-flash" # Default fallback
                try:
                    available = [m.name.replace('models/', '') for m in self._client.models.list() if 'gemini' in m.name.lower()]
                    if available:
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
                
                print("DEBUG SERVICE: Gemini SDK Client initialized as Fallback.")
                return

            print("WARNING: Neither OPENAI_API_KEY nor GEMINI_API_KEY were found. AI features will be disabled.")
            
        except Exception as e:
            print(f"Error initializing AI Clients: {e}")

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

    def encode_image_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _validate_image_with_ai(self, image_path):
        """
        استخدام الـ API المتاح لتمييز نوع الصورة سواء OpenAI أو Gemini
        """
        if not self.ai_provider:
             print("ERROR: AI features are disabled due to missing keys.")
             return "ERROR"
             
        try:
            prompt = "Classify this image: X-RAY (if chest x-ray), SKIN (if skin lesion), or INVALID (not medical). Answer with one word only, exactly as [X-RAY, SKIN, INVALID]."
            
            if self.ai_provider == "openai":
                base64_image = self.encode_image_base64(image_path)
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self._system_instruction},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                category = response.choices[0].message.content.strip().upper()
                
            elif self.ai_provider == "gemini":
                img = Image.open(image_path)
                response = self._client.models.generate_content(
                    model=self._model_id,
                    contents=[prompt, img],
                    config=types.GenerateContentConfig(
                        system_instruction=self._system_instruction,
                        temperature=0.0
                    )
                )
                category = response.text.strip().upper()

            # التنظيف في حال كانت الإجابة تحتوي على نقاط أو كلمات إضافية
            for word in ["X-RAY", "SKIN", "INVALID"]:
                if word in category:
                    print(f"DEBUG VISION: {self.ai_provider.upper()} classified as -> {word}")
                    return word
            return "INVALID"
        except Exception as e:
            print(f"Vision Client Error ({self.ai_provider}): {e}")
            return "ERROR"

    def predict_pneumonia(self, image_path):
        start_time = time.time()
        category = self._validate_image_with_ai(image_path)
        
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
        category = self._validate_image_with_ai(image_path)
        
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
        
        # Original model labels: ['akiec', 'bcc', 'bkl', 'df', 'mel','nv', 'vasc']
        # 0: Actinic keratoses (akiec)
        # 1: Basal cell carcinoma (bcc)
        # 2: Benign keratosis-like lesions (bkl)
        # 3: Dermatofibroma (df)
        # 4: Melanoma (mel)
        # 5: Melanocytic nevi (nv)
        # 6: Vascular lesions (vasc)
        labels = [
            "Actinic keratoses",
            "Basal cell carcinoma",
            "Benign keratosis-like lesions",
            "Dermatofibroma",
            "Melanoma",
            "Melanocytic nevi",
            "Vascular lesions"
        ]
        
        print("DEBUG: Skin Cancer Model Probabilities:")
        for i, label_name in enumerate(labels):
            print(f"Index {i} ({label_name}): {output_data[i]:.4f}")
            
        top_indices = np.argsort(output_data)[-3:][::-1]
        prob = float(output_data[top_indices[0]])
        
        raw_label = labels[top_indices[0]]
        # bkl (Benign keratosis-like lesions), df (Dermatofibroma), nv (Melanocytic nevi), vasc (Vascular lesions) are generally benign.
        benign_list = ["Melanocytic nevi", "Benign keratosis-like lesions", "Dermatofibroma", "Vascular lesions"]
        label = "سليم (التصبغات تبدو حميدة/طبيعية)" if raw_label in benign_list else f"مصاب محتمل ({raw_label})"

        print(f"DEBUG: Skin Analyzed in {time.time() - start_time:.2f}s")
        return {
            "class": label, 
            "raw_class": raw_label, 
            "probability": prob,
            "ai_advice": f"تم التعرف على المنطقة المصابة. النتيجة الأولية: {label}."
        }, prob

    def get_ai_advice(self, message, history=None):
        if not self.ai_provider:
             return "عذراً، خدمة الذكاء الاصطناعي معطلة لعدم وجود مفتاح (API Key) صالح."
             
        try:
            if self.ai_provider == "openai":
                messages = [{"role": "system", "content": self._system_instruction}]
                
                if history:
                    for entry in history:
                        role = "user" if entry.get("sender") == "user" else "assistant"
                        messages.append({"role": role, "content": entry.get("text", "")})
                
                messages.append({"role": "user", "content": message})
                
                response = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.4
                )
                return response.choices[0].message.content
                
            elif self.ai_provider == "gemini":
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
            print(f"Chat API Client Error ({self.ai_provider}): {e}")
            return "لم أستطع معالجة طلبك حالياً، يرجى المحاولة لاحقاً."
