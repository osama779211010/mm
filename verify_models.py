import os
import tensorflow as tf
import numpy as np

MODELS_ROOT = r'C:\Users\ENG\Desktop\MAS\mas\assets\models'

def verify_pneumonia():
    print("\n--- Verifying Pneumonia (CheXNet) ---")
    path = os.path.join(MODELS_ROOT, 'CheXNet-Model-Pneumonia-classification-using-Keras-main', 'brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    
    try:
        model = tf.keras.applications.DenseNet121(weights=None, classes=14)
        model.load_weights(path)
        print("Pneumonia model weights loaded successfully.")
        
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        preds = model.predict(dummy_input)
        print(f"Shape of predictions: {preds.shape}")
        print(f"Sample prediction (Index 6): {preds[0][6]}")
    except Exception as e:
        print(f"Failed to load or run Pneumonia model: {e}")

def verify_skin_cancer():
    print("\n--- Verifying Skin Cancer (TFLite) ---")
    path = os.path.join(MODELS_ROOT, 'Skin-Cancer-Classification-Tflite-Model-master', 'model.tflite')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return
    
    try:
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        print("Skin Cancer TFLite interpreter allocated successfully.")
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"Shape of output: {output_data.shape}")
        print(f"Raw output (prob): {output_data[0]}")
    except Exception as e:
        print(f"Failed to load or run Skin Cancer model: {e}")

if __name__ == "__main__":
    verify_pneumonia()
    verify_skin_cancer()
