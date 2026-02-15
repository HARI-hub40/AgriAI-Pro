import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests
import time

# 1. மாடலை கிளவுடில் இருந்து டவுன்லோட் செய்யும் பகுதி
@st.cache_resource
def load_my_model():
    model_path = 'plant_model.h5'
    # உங்க கூகுள் டிரைவ் ஐடி
    file_id = '1KajoQUALvXX_x4ZlGsR5pPlBcQAkCBME'
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    if not os.path.exists(model_path):
        with st.spinner('AI மாடல் தயாராகிறது... இது ஒரு நிமிடம் எடுக்கும்... தயவுசெய்து காத்திருக்கவும்!'):
            try:
                response = requests.get(url, stream=True)
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                time.sleep(2) # ஃபைல் சேவ் ஆக சின்ன இடைவெளி
            except Exception as e:
                st.error(f"மாடல் டவுன்லோட் செய்வதில் சிக்கல்: {e}")
    
    return tf.keras.models.load_model(model_path)

# 2. ஆப் வடிவமைப்பு
st.set_page_config(page_title="AgriAI Pro", page_icon="🌿")
st.title("🌿 AgriAI Pro - இலை நோய் கண்டறிதல்")
st.write("மொபைல் கேமரா மூலம் இலையைப் படம் பிடித்து நோயைக் கண்டறியவும்.")

try:
    # மாடலை லோட் செய்தல்
    model = load_my_model()
    
    # கேமரா இன்புட்
    img_file = st.camera_input("இலையை ஸ்கேன் செய்யவும்")

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="நீங்கள் எடுத்த படம்", use_container_width=True)
        
        # இமேஜ் ப்ராசஸிங்
        with st.spinner('ஆராய்ச்சி செய்கிறது...'):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # பலன்களைக் கணித்தல்
            predictions = model.predict(img_array)
            # உங்கள் மாடலில் உள்ள நோய்களின் பெயர்கள் (தேவைப்பட்டால் மாற்றவும்)
            class_names = ['Healthy (ஆரோக்கியமானது)', 'Powdery Mildew', 'Rust (துரு நோய்)'] 
            result = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            
            st.success(f"கண்டறியப்பட்ட முடிவு: {result}")
            st.info(f"உறுதித்தன்மை: {confidence:.2f}%")
            
except Exception as e:
    st.warning("மாடல் இன்னும் தயாராகவில்லை. 1 நிமிடம் காத்திருந்து பக்கத்தை 'Refresh' செய்யவும்.")
    st.info("குறிப்பு: கூகுள் டிரைவ் லிங்க் 'Anyone with link' செட்டிங்கில் இருப்பதை உறுதி செய்யவும்.")

st.write("---")
st.caption("Powered by AgriAI - Helping Farmers with Technology")
