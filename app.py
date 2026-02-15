import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
import time

@st.cache_resource
def load_my_model():
    model_path = 'plant_model.h5'
    file_id = '1KajoQUALvXX_x4ZlGsR5pPlBcQAkCBME'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner('AI Model தயாராகிறது... ஒரு நிமிடம் காத்திருக்கவும்...'):
            r = requests.get(url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(r.content)
            time.sleep(3) 
    
    return tf.keras.models.load_model(model_path)

st.set_page_config(page_title="AgriAI Pro", page_icon="🌿")
st.title("🌿 AgriAI Pro - இலை நோய் கண்டறிதல்")

try:
    model = load_my_model()
    img_file = st.camera_input("இலையை ஸ்கேன் செய்யவும்")

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="நீங்கள் எடுத்த படம்", use_container_width=True)
        
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array)
        class_names = ['Healthy (ஆரோக்கியமானது)', 'Powdery Mildew', 'Rust (துரு நோய்)'] 
        result = class_names[np.argmax(predictions)]
        
        st.success(f"கண்டறியப்பட்ட முடிவு: {result}")
except Exception as e:
    st.error("காத்திருக்கவும்... மாடல் இன்னும் முழுமையாகத் தயாராகவில்லை. தயவுசெய்து 1 நிமிடம் கழித்து பக்கத்தை 'Refresh' செய்யவும்.")
