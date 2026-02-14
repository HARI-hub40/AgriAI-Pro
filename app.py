import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
from fpdf import FPDF

# 1. Page Config (Corporate Look)
st.set_page_config(page_title="AgriAI Pro - Live Diagnostic", layout="wide", page_icon="🌿")

# 2. CSS for Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #2E7D32; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('plant_model.h5')

model = load_my_model()
categories = ['Apple', 'Cherry', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Raspberry', 'Strawberry', 'Tomato_Bacterial', 'Tomato_Early_Blight', 'Tomato_Late_Blight', 'Tomato_Healthy']

# 4. Sidebar - Settings
with st.sidebar:
    st.title("🛡️ Control Center")
    lang = st.radio("Language / மொழி", ("English", "தமிழ்"))
    mode = st.radio("Input Mode", ("Live Camera (Live Scanner)", "Upload Image (File)"))
    st.info("System Status: Active ✅")

# 5. Main UI
st.title("🌿 AgriAI Pro: Next-Gen Plant Diagnostic System")
st.write("Professional Real-time Analysis for Modern Agriculture.")
st.markdown("---")

input_img = None

if mode == "Live Camera (Live Scanner)":
    # ரியல்-டைம் கேமரா வசதி
    input_img = st.camera_input("இலையை கேமராவிற்கு நேராக காட்டவும் (Show Leaf to Camera)")
else:
    input_img = st.file_uploader("Upload Leaf Photo", type=["jpg", "png", "jpeg"])

if input_img is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Scanning Input")
        image = Image.open(input_img)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("📊 AI Analysis Report")
        with st.spinner('Neural Network Processing...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediction
            prediction = model.predict(img_array)
            result_idx = np.argmax(prediction)
            result = categories[result_idx]
            conf = np.max(prediction) * 100

            # Display Results
            if lang == "English":
                st.metric("Detected Status", result, f"{conf:.2f}% Confidence")
                st.success(f"**Action Required:** Monitor and apply appropriate treatment.")
            else:
                st.metric("கண்டறியப்பட்ட நிலை", result, f"{conf:.2f}% துல்லியம்")
                st.warning(f"**பரிந்துரை:** பாதிக்கப்பட்ட பகுதியை கவனித்து தேவையான மருந்து தெளிக்கவும்.")

            st.progress(int(conf))

            # 6. PDF Report Generation (The Pro Feature)
            if st.button("Generate & Download PDF Report"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(200, 10, txt="AgriAI Pro Diagnostic Report", ln=True, align='C')
                pdf.set_font("Arial", size=12)
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.cell(200, 10, txt=f"Detected Species/Disease: {result}", ln=True)
                pdf.cell(200, 10, txt=f"Confidence Level: {conf:.2f}%", ln=True)
                pdf.ln(10)
                pdf.multi_cell(0, 10, txt="Disclaimer: This is an AI-generated report. Please consult an agricultural expert for critical decisions.")
                
                pdf_output = "AgriAI_Report.pdf"
                pdf.output(pdf_output)
                
                with open(pdf_output, "rb") as f:
                    st.download_button("Click to Download PDF", f, file_name="AgriAI_Report.pdf")

else:
    st.info("மாப்ள, மேல இருக்குற கேமராவை ஆன் பண்ணுங்க அல்லது போட்டோவை அப்லோட் பண்ணுங்க!")

st.markdown("---")
st.caption("© 2026 AgriAI Technologies | Hardware: Victus RTX 3050")