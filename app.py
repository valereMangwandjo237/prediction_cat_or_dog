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
    st.image(image, caption="Image chargée", use_column_width=True)

    # Prétraiter l'image
    image = image.resize((128, 128))  # Redimensionner comme pour le modèle
    image = np.array(image) / 255.0   # Normaliser
    image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch

    st.write(f"Image shape: {image.shape}")

    # Faire la prédiction
    prediction = model.predict(image)[0][0]
    if prediction >= 0.5:
        st.write("C'est un chien 🐶")
    else:
        st.write("C'est un chat 🐱")
