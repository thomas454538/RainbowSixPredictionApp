import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import os

# Page Configuration
st.set_page_config(
    page_title="Prédiction des Wins",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load and prepare data
dataTomClancy = pd.read_csv('./rs6_clean.csv')
colonnes = ['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played', 'wins']
GoodDataTomClancy = dataTomClancy[colonnes]

# Calculate medians for display
medians = GoodDataTomClancy[['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played']].median()

# Function to load the model with caching
@st.cache_resource
def load_model():
    model_filename = 'random_forest_model.joblib'
    if not os.path.isfile(model_filename):
        st.error(f"Le fichier {model_filename} n'existe pas dans le répertoire racine.")
        st.stop()
    # Load the model
    model = load(model_filename)
    return model

try:
    model = load_model()
    st.write("Le modèle a été chargé avec succès.")
except ValueError as e:
    st.error("Erreur lors du chargement du modèle : incompatibilité de versions. Veuillez vérifier la version de scikit-learn utilisée pour enregistrer le modèle.")
    st.write("Détails de l'erreur :", e)
    st.stop()
except Exception as e:
    st.error("Erreur inattendue lors du chargement du modèle.")
    st.write("Détails de l'erreur :", e)
    st.stop()

# Feature labels
feature_labels = {
    'kills': 'Nombre de kills',
    'deaths': 'Nombre de deaths',
    'losess': 'Nombre de losses',
    'xp': 'Nombre de XP',
    'headshots': 'Nombre de headshots',
    'games_played': 'Nombre de parties jouées',
    'time_played': 'Temps joué (en secondes)'
}

# Streamlit User Interface
st.title("🎮 Prédiction des Wins")
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("### Entrez les caractéristiques du joueur pour prédire si le nombre de wins est supérieur à la médiane")
st.markdown("<br>", unsafe_allow_html=True)

# Collect user input
def get_user_input(medians):
    user_input = {}
    col1, col2 = st.columns(2)
    features = list(medians.index)
    for i, feature in enumerate(features):
        label = feature_labels.get(feature, feature)
        default_value = int(medians[feature])
        help_text = 'Temps joué en secondes' if feature == 'time_played' else None

        if i % 2 == 0:
            with col1:
                value = st.number_input(label, min_value=0, value=default_value, help=help_text)
        else:
            with col2:
                value = st.number_input(label, min_value=0, value=default_value, help=help_text)
        user_input[feature] = value
    return pd.DataFrame([user_input])

# Prepare user input for prediction
donnees_utilisateur = get_user_input(medians)

# Apply custom CSS styling
st.markdown("""
    <style>
    /* Center the main content */
    .main {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    /* Light background */
    body {
        background: linear-gradient(135deg, #ffffff, #f0f4f8);
        color: #333333;
    }
    /* General interface styling */
    .stApp {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    /* Title styling */
    h1 {
        color: #333333;
        text-align: center;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 10px 20px;
        transition: transform 0.2s ease, background-color 0.2s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #45a049;
    }
    /* Input fields */
    input {
        background-color: #f0f4f8;
        color: #333333;
        border-radius: 4px;
        padding: 10px;
        border: 1px solid #cccccc;
    }
    /* Success and error messages styling */
    .stAlert {
        font-size: 18px;
    }
    /* Hide footer and header */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Ensure donnees_utilisateur has correct columns for model
donnees_utilisateur = donnees_utilisateur.reindex(columns=model.feature_names_in_, fill_value=0)

# Prediction button and results display
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Prédire"):
    try:
        with st.spinner('Calcul de la prédiction... '):
            prediction = model.predict(donnees_utilisateur)
        
        if prediction[0] == 1:
            st.success("🎉 Prédiction : Le nombre de 'wins' est probablement **au-dessus** de la médiane.")
        else:
            st.warning("❌ Prédiction : Le nombre de 'wins' est probablement **en-dessous** de la médiane.")
    
    except Exception as e:
        # Debug information for troubleshooting
        st.write("Détails de l'erreur donnees_utilisateur.values :", donnees_utilisateur.values)
        st.write("Détails de l'erreur model.feature_names_in_ :", model.feature_names_in_)
        st.error("Une erreur est survenue lors de la prédiction.")
        st.write("Détails de l'erreur :", e)
