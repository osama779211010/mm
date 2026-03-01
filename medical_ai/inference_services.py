import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms

class AIInferenceService:
    def __init__(self):
        # استخدام مسار نسبي للموديلات لضمان العمل على السيرفر (Cloud Deployment)
        self.models_root = os.path.join(os.path.dirname(__file__), 'models')
        
        # تحميل الموديلات عند الطلب لتوفير الذاكرة (Lazy Loading)
        self._pneumonia_model = None
        self._skin_cancer_interpreter = None
        self._brain_tumor_model = None
        self._gemma_model = None

    @property
    def pneumonia_model(self):
        if self._pneumonia_model is None:
            self._pneumonia_model = self._load_pneumonia_model()
        return self._pneumonia_model

    @property
    def skin_cancer_interpreter(self):
        if self._skin_cancer_interpreter is None:
            self._skin_cancer_interpreter = self._load_skin_cancer_model()
        return self._skin_cancer_interpreter

    def _load_pneumonia_model(self):
        try:
            # استخدام موديل TFLite المحسن للجوال (MobileNetV2)
            path = os.path.join(self.models_root, 'pneumo', 'pneumonia_model.tflite')
            if os.path.exists(path):
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                print(f"DEBUG SERVICE: Pneumonia TFLite model loaded from {path}")
                return interpreter
        except Exception as e:
            print(f"Error loading Pneumonia model: {e}")
        return None

    @property
    def pneumonia_interpreter(self):
        if self._pneumonia_model is None:
            self._pneumonia_model = self._load_pneumonia_model()
        return self._pneumonia_model

    def _load_skin_cancer_model(self):
        try:
            path = os.path.join(self.models_root, 'Skin-Cancer-Classification-Tflite-Model-master', 'model.tflite')
            if os.path.exists(path):
                interpreter = tf.lite.Interpreter(model_path=path)
                interpreter.allocate_tensors()
                return interpreter
        except Exception as e:
            print(f"Error loading Skin Cancer model: {e}")
        return None

    def _is_valid_medical_image(self, img_array, diag_type):
        """
        التحقق البسيط من أن الصورة ذات صلة بالتشخيص الطبي المطلوب.
        """
        # 1. التحقق من التباين والوضوح
        std_dev = np.std(img_array)
        if std_dev < 0.02:
            return False, "الصورة غير واضحة أو باهتة جداً، يرجى إعادة محاولة التصوير."

        # 2. التحقق من الأشعة السينية (غالبية الأشعة تكون بتدرج رمادي)
        if diag_type == 'PNEUMONIA':
            # تحويل مصفوفة [0,1] إلى [0,255] للتحليل اللوني
            img_uint8 = (img_array * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1].mean()
            # إذا كانت الألوان مشبعة جداً، فهي غالباً ليست أشعة صدرية سينية
            if saturation > 45:
                return False, "هذه الصورة ملونة جداً، يبدو أنها ليست أشعة سينية للصدر. يرجى رفع صورة أشعة صدر صحيحة."
        
        # 3. التحقق من صور الجلدية (تحتاج لفحص أدق لاحقاً)
        if diag_type == 'SKIN_CANCER':
            # يمكنك إضافة فحص لألوان الجلد هنا، لكن سنكتفي حالياً بدقة الموديل
            pass

        return True, ""

    def predict_pneumonia(self, image_path):
        print(f"DEBUG SERVICE: Starting Pneumonia TFLite prediction for {image_path}")
        interpreter = self.pneumonia_interpreter
        if not interpreter:
            return {"error": "Model not loaded"}, 0.0

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # تجهيز الصورة: 224x224 RGB مع تطبيع [0, 1]
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # التحقق من نوع الصورة
        is_valid, error_msg = self._is_valid_medical_image(img_array, 'PNEUMONIA')
        if not is_valid:
            print(f"DEBUG SERVICE: Invalid image detected - {error_msg}")
            return {"error": "INVALID_IMAGE", "message": error_msg, "class": "Invalid Input"}, 0.0

        img_batch = np.expand_dims(img_array, axis=0)
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(output_details[0]['index'])
        # الموديل يعطي قيمة واحدة (sigmoid) حيث > 0.5 تعني التهاب رئوي
        p_prob = float(preds[0][0])
        print(f"DEBUG SERVICE: Pneumonia Raw Prob: {p_prob}")
        
        if p_prob > 0.5:
            label = "Pneumonia"
            display_conf = p_prob
        else:
            label = "Normal"
            display_conf = 1.0 - p_prob

        return {"class": label, "probability": p_prob}, display_conf

    def predict_skin_cancer(self, image_path):
        print(f"DEBUG SERVICE: Starting Skin Cancer prediction for {image_path}")
        if not self.skin_cancer_interpreter:
             return {"error": "Model not loaded"}, 0.0

        input_details = self.skin_cancer_interpreter.get_input_details()
        output_details = self.skin_cancer_interpreter.get_output_details()

        # التأكد من التحويل لـ RGB لحل مشكلة الـ alpha channel (RGBA)
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        # استخدام معالجة MobileNet (Scaling pixels to [-1, 1])
        img_array = np.array(img, dtype=np.float32)
        
        # التحقق الأولي
        is_valid, error_msg = self._is_valid_medical_image(img_array / 255.0, 'SKIN_CANCER')
        if not is_valid:
            return {"error": "INVALID_IMAGE", "message": error_msg, "class": "Invalid Input"}, 0.0

        input_data = np.expand_dims((img_array / 127.5) - 1.0, axis=0)

        self.skin_cancer_interpreter.set_tensor(input_details[0]['index'], input_data)
        self.skin_cancer_interpreter.invoke()
        output_data = self.skin_cancer_interpreter.get_tensor(output_details[0]['index'])[0]
        print(f"DEBUG SERVICE: Skin Cancer Raw Probs: {output_data}")
        
        # تصحيح ترتيب التسميات بناءً على تدريب الموديل (class_indices)
        # {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
        labels = [
            "Actinic keratoses",          # akiec
            "Basal cell carcinoma",       # bcc
            "Benign keratosis-like lesions", # bkl
            "Dermatofibroma",             # df
            "Melanoma",                   # mel
            "Melanocytic nevi",           # nv
            "Vascular lesions"            # vasc
        ]
        
        # الحصول على أعلى 3 توقعات
        top_indices = np.argsort(output_data)[-3:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "label": labels[idx],
                "probability": float(output_data[idx])
            })
            
        prob = float(output_data[top_indices[0]])
        raw_label = labels[top_indices[0]]
        
        # تصنيف الحالات الحميدة (nv, bkl, df, vasc) كـ Normal/Benign
        benign_labels = ["Melanocytic nevi", "Benign keratosis-like lesions", "Dermatofibroma", "Vascular lesions"]
        
        if raw_label in benign_labels:
            label = "Normal/Benign"
        else:
            label = raw_label
            
        return {
            "class": label, 
            "raw_class": raw_label, 
            "probability": prob,
            "top_3": results
        }, prob

    def predict_brain_tumor(self, image_path):
        print(f"DEBUG SERVICE: Starting Brain Tumor prediction (Mock) for {image_path}")
        return {"class": "Stable", "probability": 0.95}, 0.95

    def get_ai_advice(self, message):
        return "هذه نصيحة استرشادية بناءً على البيانات المتوفرة. يرجى مراجعة الطبيب."
