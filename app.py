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
    # நேரடி டவுன்லோட் லிங்க்
    file_id = '1KajoQUALvXX_x4ZlGsR5pPlBcQAkCBME'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner('AI மாடல் தயாராகிறது... தயவுசெய்து 2 நிமிடம் காத்திருக்கவும்...'):
            r = requests.get(url, allow_redirects=True)
            with open(model_path, 'wb') as f:
                f.write(r.content)
            # ஃபைல் முழுசா சேவ் ஆக ஒரு 5 செகண்ட் எக்ஸ்ட்ரா டைம்
            time.sleep(5)
    
    # ஃபைல் இருக்கான்னு செக் பண்ணிட்டு லோட் பண்ணும்
    if os.path.getsize(model_path) > 0:
        return tf.keras.models.load_model(model_path)
    else:
        st.error("மாடல் ஃபைல் சரியாக டவுன்லோட் ஆகவில்லை.")
        return None

st.title("🌿 AgriAI Pro")

try:
    model = load_my_model()
    if model:
        img_file = st.camera_input("இலையை ஸ்கேன் செய்யவும்")
        if img_file:
            image = Image.open(img_file)
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            class_names = ['Healthy', 'Powdery Mildew', 'Rust'] 
            st.success(f"முடிவு: {class_names[np.argmax(predictions)]}")
except Exception as e:
    st.error(f"காத்திருக்கவும்... மாடல் இன்னும் முழுமையாகப் பதிவிறக்கம் ஆகவில்லை. 1 நிமிடம் கழித்து Refresh செய்யவும்.")
