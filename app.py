import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# 1. மாடலை டவுன்லோட் செய்யும் பகுதி
@st.cache_resource
def load_my_model():
    model_path = 'plant_model.h5'
    file_id = '1KajoQUALvXX_x4ZlGsR5pPlBcQAkCBME'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner('AI Model தயாராகிறது... தயவுசெய்து காத்திருக்கவும்...'):
            r = requests.get(url, allow_redirects=True)
            open(model_path, 'wb').write(r.content)
    
    return tf.keras.models.load_model(model_path)

# 2. ஆப் வடிவமைப்பு
st.set_page_config(page_title="AgriAI Pro", page_icon="🌿")
st.title("🌿 AgriAI Pro - இலை நோய் கண்டறிதல்")

try:
    model = load_my_model()
    
    img_file = st.camera_input("இலையை ஒரு போட்டோ எடுக்கவும்")

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="நீங்கள் எடுத்த படம்", use_container_width=True)
        
        # படத்தைச் சரிசெய்தல்
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # கணிப்பு
        predictions = model.predict(img_array)
        class_names = ['Healthy', 'Powdery Mildew', 'Rust'] 
        result = class_names[np.argmax(predictions)]
        
        st.success(f"கண்டறியப்பட்ட முடிவு: {result}")
        
except Exception as e:
    st.error(f"எரர் வந்துள்ளது: {e}")
