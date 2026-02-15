import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# 1. மேகக்கணியில் (Cloud) இருந்து மாடல் கோப்பை பதிவிறக்கம் செய்யும் செயல்பாடு
@st.cache_resource
def load_my_model():
    model_path = 'plant_model.h5'
    # உங்கள் கூகுள் டிரைவ் கோப்பு ஐடி
    file_id = '1KajoQUALvXX_x4ZlGsR5pPlBcQAkCBME'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner('Downloading AI Model from Cloud... Please wait...'):
            try:
                response = requests.get(url)
                with open(model_path, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
    
    return tf.keras.models.load_model(model_path)

# மாடலை ஏற்றவும்
model = load_my_model()

# 2. தாவர நோய் பெயர்கள் (உங்க மாடலுக்கு ஏற்ப இதை மாற்றிக்கொள்ளலாம்)
class_names = ['Healthy', 'Powdery Mildew', 'Rust'] 

# 3. ஆப் இடைமுகம் (User Interface)
st.set_page_config(page_title="AgriAI Pro", layout="centered")
st.title("🌿 AgriAI Pro - Plant Disease Detector")
st.write("Take a photo or upload an image of the leaf to identify diseases.")

# மொபைல் கேமரா அல்லது கேலரி மூலம் படம் எடுத்தல்
img_file = st.camera_input("Scan Leaf")

if img_file is not None:
    # படத்தை காண்பிக்கவும்
    image = Image.open(img_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # படத்தை ஏஐ மாடலுக்கு தயார் செய்தல்
    with st.spinner('Analyzing...'):
        img = image.resize((224, 224)) # உங்கள் மாடல் அளவுக்கு ஏற்ப மாற்றவும்
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # கணிப்பு (Prediction)
        predictions = model.predict(img_array)
        result = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # முடிவை காட்டுதல்
        st.success(f"Result: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

st.write("---")
st.caption("Powered by AgriAI Cloud Technology")
