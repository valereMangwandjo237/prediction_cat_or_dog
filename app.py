import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Charger le modèle entraîné
model = load_model("model_cat_dog.keras")  # Assure-toi que le modèle est dans le même dossier

# Définir les classes (ex: 0 = Chat, 1 = Chien)
classes = ["Chat", "Chien"]

# Titre de l'application
st.title("🐶🐱 Détection Chat vs Chien")

# Interface pour uploader une image
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée",  use_container_width=True)
    if st.button("Predire l'image", help="Cliquez pour envoyer les donnees", type="primary"):
        st.write("En cours...")
