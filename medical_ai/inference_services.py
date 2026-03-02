import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

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
        التحقق من أن الصورة ذات صلة بالتشخيص الطبي المطلوب.
        """
        # 1. التحقق من التباين والوضوح
        std_dev = np.std(img_array)
        print(f"DEBUG VALIDATION: std_dev={std_dev:.5f}")
        if std_dev < 0.01: # تم تقليل الحساسية قليلاً من 0.015
            return False, "الصورة غير واضحة تماماً أو باهتة، يرجى إعادة محاولة التصوير في إضاءة جيدة."

        # 2. التحقق من كثافة الحواف (Edge Density)
        img_uint8 = (img_array * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150) # تقليل عتبة كاني ليكون أكثر حساسية
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        print(f"DEBUG VALIDATION: edge_density={edge_density:.5f}")
        if edge_density < 0.002: # تم تقليلها من 0.005 لضمان قبول الأشعة السلسة
            return False, "الصورة تبدو فارغة أو لا تحتوي على تفاصيل كافية للتحليل."

        # 3. التحقق من صور الأشعة (PNEUMONIA)
        if diag_type == 'PNEUMONIA':
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            saturation = hsv[:,:,1].mean()
            print(f"DEBUG VALIDATION [X-RAY]: saturation={saturation:.2f}")
            # زيادة الحد المسموح به للتشبع اللوني لضمان قبول الأشعة المصورة بكاميرا هاتف
            if saturation > 60: # تم رفعها من 35
                return False, "هذه الصورة تحتوي على ألوان مشبعة جداً، يبدو أنها ليست أشعة سينية. يرجى رفع صورة أشعة صدر باللونين الأبيض والأسود."
            
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            diversity = np.count_nonzero(hist > (gray.size * 0.005)) # تقليل النسبة المئوية للعد
            print(f"DEBUG VALIDATION [X-RAY]: histogram diversity={diversity}")
            if diversity < 15: # تم تقليلها من 30
                return False, "الصورة لا تحتوي على تدرج رمادي كافٍ للأشعة. يرجى رفع صورة واضحة."
        
        # 4. التحقق من صور الجلد (SKIN_CANCER)
        if diag_type == 'SKIN_CANCER':
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            # توسيع نطاق لون البشرة قليلاً
            lower_skin = np.array([0, 10, 40], dtype=np.uint8)
            upper_skin = np.array([40, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
            print(f"DEBUG VALIDATION [SKIN]: skin_ratio={skin_ratio:.3f}")
            
            if skin_ratio < 0.04: # تم تقليلها من 0.10 لتناسب لقطات الجلد المقربة
                return False, "الرجاء رفع صورة واضحة لمنطقة الجلد المصابة. لم يتم التعرف على ملامح بشرة كافية."

        return True, ""

    def predict_pneumonia(self, image_path):
        print(f"DEBUG SERVICE: Starting Pneumonia TFLite prediction for {image_path}")
        interpreter = self.pneumonia_interpreter
        if not interpreter:
            return {"error": "Model not loaded"}, 0.0

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        is_valid, error_msg = self._is_valid_medical_image(img_array, 'PNEUMONIA')
        if not is_valid:
            return {"error": "INVALID_IMAGE", "message": error_msg, "class": "بيانات غير صالحة"}, 0.0

        img_batch = np.expand_dims(img_array, axis=0)
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        preds = interpreter.get_tensor(output_details[0]['index'])
        p_prob = float(preds[0][0])
        
        if 0.49 < p_prob < 0.51: # تم تضييق نطاق الشك من 0.45-0.55
            return {"error": "INVALID_IMAGE", "message": "النتيجة غير حاسمة، يرجى رفع صورة أشعة أكثر دقة.", "class": "غير مؤكد"}, 0.0

        if p_prob > 0.5:
            label = "مصاب (Pneumonia)"
            display_conf = p_prob
        else:
            label = "غير مصاب (Normal)"
            display_conf = 1.0 - p_prob

        return {"class": label, "probability": p_prob}, display_conf

    def predict_skin_cancer(self, image_path):
        print(f"DEBUG SERVICE: Starting Skin Cancer prediction for {image_path}")
        if not self.skin_cancer_interpreter:
             return {"error": "Model not loaded"}, 0.0

        input_details = self.skin_cancer_interpreter.get_input_details()
        output_details = self.skin_cancer_interpreter.get_output_details()

        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        
        is_valid, error_msg = self._is_valid_medical_image(img_array / 255.0, 'SKIN_CANCER')
        if not is_valid:
            return {"error": "INVALID_IMAGE", "message": error_msg, "class": "بيانات غير صالحة"}, 0.0

        input_data = np.expand_dims((img_array / 127.5) - 1.0, axis=0)

        self.skin_cancer_interpreter.set_tensor(input_details[0]['index'], input_data)
        self.skin_cancer_interpreter.invoke()
        output_data = self.skin_cancer_interpreter.get_tensor(output_details[0]['index'])[0]
        
        labels = [
            "Actinic keratoses",          # akiec
            "Basal cell carcinoma",       # bcc
            "Benign keratosis-like lesions", # bkl
            "Dermatofibroma",             # df
            "Melanoma",                   # mel
            "Melanocytic nevi",           # nv
            "Vascular lesions"            # vasc
        ]
        
        top_indices = np.argsort(output_data)[-3:][::-1]
        prob = float(output_data[top_indices[0]])
        
        if prob < 0.20: # تم تقليلها من 0.30
            return {"error": "INVALID_IMAGE", "message": "الدقة منخفضة جداً، يرجى تصوير المنطقة بوضوح أكبر.", "class": "غير مؤكد"}, prob

        results = []
        for idx in top_indices:
            results.append({
                "label": labels[idx],
                "probability": float(output_data[idx])
            })
            
        raw_label = labels[top_indices[0]]
        benign_labels = ["Melanocytic nevi", "Benign keratosis-like lesions", "Dermatofibroma", "Vascular lesions"]
        
        if raw_label in benign_labels:
            label = "غير مصاب (حميد/طبيعي)"
        else:
            label = f"مصاب محتمل ({raw_label})"
            
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
