import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import google.generativeai as genai
import time

class AIInferenceService:
    def __init__(self):
        # استخدام مسار نسبي للموديلات لضمان العمل على السيرفر
        self.models_root = os.path.join(os.path.dirname(__file__), 'models')
        
        # تحميل الموديلات بشكل مسبق (Pre-loading) لتسريع الاستجابة الأولى
        self._pneumonia_model = None
        self._skin_cancer_interpreter = None
        
        # إعداد Gemini
        self._setup_gemini()
        
    def _setup_gemini(self):
        try:
            api_key = "AIzaSyCazm5F_m8VWhPq9ZPIKmXc8g5TDPt_kpI"
            genai.configure(api_key=api_key)
            
            # تعليمات محسنة وشاملة للمساعد
            system_instruction = """
            أنت "MASA" (اختصار لـ Medical AI Smart Assistant)، المساعد الطبي الذكي الشامل لنظام MAS AI HUB.
            
            مهمتك الأساسية:
            1. تمييز الصور الطبية (Vision): عند استلام صورة، حدد بدقة ما إذا كانت أشعة سينية للصدر (X-RAY)، أو صورة لآفة جلدية (SKIN)، أو صورة غير طبية (INVALID).
            2. تقديم معلومات طبية وصحية دقيقة وشاملة للمستخدمين في كافة المجالات (أعراض، أدوية، تغذية، صحة نفسية).
            
            الاستجابة لتمييز الصور:
            عندما نرسل لك صورة للتمييز، أجب بكلمة واحدة فقط من هذه الخيارات: [X-RAY, SKIN, INVALID].
            
            شخصيتك:
            خبير طبي ودود، متعاطف، وبليغ. رحب بالمستخدم دائماً ووجهه للفحص السريري عند الضرورة.
            """
            
            self._gemini_model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"temperature": 0.2},
                system_instruction=system_instruction
            )
            print("DEBUG SERVICE: Gemini AI (Vision Enabled) initialized.")
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            self._gemini_model = None

    @property
    def pneumonia_interpreter(self):
        if self._pneumonia_model is None:
            path = os.path.join(self.models_root, 'pneumo', 'pneumonia_model.tflite')
            if os.path.exists(path):
                self._pneumonia_model = tf.lite.Interpreter(model_path=path)
                self._pneumonia_model.allocate_tensors()
                print(f"DEBUG SERVICE: Pneumonia model pre-loaded.")
        return self._pneumonia_model

    @property
    def skin_cancer_interpreter(self):
        if self._skin_cancer_interpreter is None:
            path = os.path.join(self.models_root, 'Skin-Cancer-Classification-Tflite-Model-master', 'model.tflite')
            if os.path.exists(path):
                self._skin_cancer_interpreter = tf.lite.Interpreter(model_path=path)
                self._skin_cancer_interpreter.allocate_tensors()
                print(f"DEBUG SERVICE: Skin Cancer model pre-loaded.")
        return self._skin_cancer_interpreter

    def _distinguish_with_gemini(self, image_path):
        """
        استخدام Gemini لتمييز نوع الصورة طبياً.
        """
        if not self._gemini_model:
            return "UNKNOWN"
            
        try:
            img = Image.open(image_path)
            prompt = "Classify this image into one of these categories: X-RAY (if it's a chest x-ray), SKIN (if it's a skin lesion or skin part), or INVALID (if it's not a medical related image or not clear). Answer with the category name only."
            
            response = self._gemini_model.generate_content([prompt, img])
            category = response.text.strip().upper()
            print(f"DEBUG VISION: Gemini classified image as -> {category}")
            return category
        except Exception as e:
            print(f"Vision Error: {e}")
            return "ERROR"

    def predict_pneumonia(self, image_path):
        start_time = time.time()
        
        # الخطوة 1: التمييز الذكي باستخدام Gemini
        category = self._distinguish_with_gemini(image_path)
        if category != "X-RAY":
            return {"error": "INVALID_IMAGE", "message": "الصورة المرفوعة لا تبدو كأشعة سينية للصدر. يرجى التأكد من رفع الصورة الصحيحة.", "class": "غير صالحة"}, 0.0

        # الخطوة 2: التحليل التخصصي (سريع جداً باستخدام TFLite)
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
        
        print(f"DEBUG PERFORMANCE: Prediction took {time.time() - start_time:.2f}s")
        return {
            "class": label, 
            "probability": p_prob, 
            "ai_advice": "تم التحقق من نوع الصورة بواسطة الذكاء الاصطناعي بنجاح. " + 
                         ("الرئة تظهر علامات التهاب." if p_prob > 0.5 else "الرئة تظهر بشكل طبيعي.")
        }, display_conf

    def predict_skin_cancer(self, image_path):
        start_time = time.time()
        
        # الخطوة 1: التمييز الذكي
        category = self._distinguish_with_gemini(image_path)
        if category != "SKIN":
            return {"error": "INVALID_IMAGE", "message": "الصورة المرفوعة لا تبدو كآفة جلدية. يرجى تصوير المنطقة المصابة بوضوح.", "class": "غير صالحة"}, 0.0

        # الخطوة 2: التحليل التخصصي
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

        print(f"DEBUG PERFORMANCE: Skin prediction took {time.time() - start_time:.2f}s")
        return {
            "class": label, 
            "raw_class": raw_label, 
            "probability": prob,
            "ai_advice": f"تم التعرف على الصورة كآفة جلدية. النتيجة الأولية: {label}."
        }, prob

    def get_ai_advice(self, message, history=None):
        if not self._gemini_model:
            return "عذراً، خدمة الذكاء الاصطناعي غير متوفرة حالياً."
            
        try:
            chat_history = []
            if history:
                for entry in history:
                    role = "user" if entry.get("sender") == "user" else "model"
                    chat_history.append({"role": role, "parts": [entry.get("text", "")]})
            
            chat_session = self._gemini_model.start_chat(history=chat_history)
            response = chat_session.send_message(message)
            return response.text
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return "حدث خطأ أثناء معالجة طلبك المعذرة."
