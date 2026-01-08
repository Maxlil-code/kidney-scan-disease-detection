import streamlit as st
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

st.set_page_config(page_title="Détection Tumeurs Rénales", layout="centered")

st.title("Détection de Tumeurs Rénales (CT Scan)")

model_path = "modele_rein_vgg16_lite.h5"


# Charger le modèle avec cache pour éviter de le recharger à chaque interaction
@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        try:
            file_id = '1mteXbbOIGk63fffu-xXZgC-M-_b-PEOB'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
        except Exception as e:
            st.error(f"Erreur lors du téléchargement depuis Google Drive : {e}")
            st.stop()
    
    return tf.keras.models.load_model(model_path)

try:
    model = load_model()
    st.success("Modèle chargé avec succès")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Section upload
st.subheader("Télécharger une image CT")
uploaded_file = st.file_uploader("Choisissez une image (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Variable pour l'image affichée
img_display = None

if uploaded_file is not None:
    img_display = Image.open(uploaded_file)
    st.image(img_display, caption="Image uploadée", use_column_width=True)
    
    if st.button("Analyser l'image"):
        with st.spinner("Analyse en cours..."):
            try:
                img_resized = img_display.resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prétraitement VGG16
                img_preprocessed = preprocess_input(img_array)
                
                # Prédiction
                prediction = model.predict(img_preprocessed, verbose=0)
                probability = prediction[0][0]
                
                st.divider()
                st.subheader("Résultats de l'analyse")
                
                if probability > 0.5:
                    st.markdown(
                        f"<h2 style='color: red; text-align: center;'>Attention : Tumeur détectée</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='font-size: 18px; text-align: center;'>Confiance: {probability*100:.2f}%</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<h2 style='color: green; text-align: center;'>Analyse : Calcul Rénal (Stone)</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='font-size: 18px; text-align: center;'>Confiance: {(1-probability)*100:.2f}%</p>",
                        unsafe_allow_html=True
                    )
                
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilité Tumeur", f"{probability*100:.1f}%")
                with col2:
                    st.metric("Probabilité Stone", f"{(1-probability)*100:.1f}%")
                    
            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")
else:
    st.info("Veuillez télécharger une image CT pour commencer l'analyse")

st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px;'>Application de diagnostic assisté par IA - Modèle VGG16</p>",
    unsafe_allow_html=True
)
