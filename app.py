import streamlit as st
import numpy as np
import pandas as pd
import requests
from joblib import load

# Configuration de la page
st.set_page_config(page_title="Prédiction Wins", page_icon="🎮", layout="centered", initial_sidebar_state="collapsed")

# Chargement et préparation des données
dataTomClancy = pd.read_csv('./rs6_clean.csv')
colonnes = ['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played', 'wins']
GoodDataTomClancy = dataTomClancy[colonnes]


# URL pour le modèle
url = 'https://raw.githubusercontent.com/thomas454538/RainbowSixPredictionApp/main/ensemble_trees.joblib'

try:
    # Télécharger le modèle
    response = requests.get(url)
    with open('ensemble_trees.joblib', 'wb') as f:
        f.write(response.content)
    
    # Charger le modèle
    trees = load('ensemble_trees.joblib')
    st.write("Le modèle d'ensemble a été chargé avec succès.")
    
except ValueError as e:
    st.error("Erreur lors du chargement du modèle : incompatibilité de versions. Veuillez vérifier la version de scikit-learn utilisée pour enregistrer le modèle.")
    st.write("Détails de l'erreur :", e)
except Exception as e:
    st.error("Erreur inattendue lors du chargement du modèle.")
    st.write("Détails de l'erreur :", e)

n_estimators = len(trees)

# Calcul des médianes pour affichage
medians = GoodDataTomClancy[['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played']].median()

# Interface utilisateur Streamlit
st.title("🎮 Prédiction des Wins")
st.markdown("### Entrez les caractéristiques du joueur pour prédire si le nombre de wins est supérieur à la médiane")

# Style personnalisé avec CSS
st.markdown("""
    <style>
    /* Arrière-plan clair */
    body {
        background: linear-gradient(135deg, #ffffff, #f0f4f8);
        color: #333333;
    }

    /* Style général de l'interface */
    .stApp {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 30px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Bouton interactif */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: 1px solid #4CAF50;
        border-radius: 8px;
        padding: 10px 20px;
        transition: transform 0.2s ease, background-color 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        background-color: #45a049;
    }

    /* Champs de saisie */
    input {
        background-color: #f0f4f8;
        color: #333333;
        border-radius: 4px;
        padding: 10px;
        border: 1px solid #cccccc;
    }
    
    </style>
""", unsafe_allow_html=True)

# Barre latérale avec les valeurs médianes
st.sidebar.header("Valeurs médianes pour chaque caractéristique")
for feature, median_value in medians.items():
    st.sidebar.write(f"{feature.capitalize()} : {int(median_value)}")

# Champs de saisie pour les caractéristiques utilisateur
st.subheader("Caractéristiques")
kills = st.number_input("Nombre de kills", min_value=0, value=int(medians['kills']))
deaths = st.number_input("Nombre de deaths", min_value=0, value=int(medians['deaths']))
losess = st.number_input("Nombre de losess", min_value=0, value=int(medians['losess']))
xp = st.number_input("Nombre de XP", min_value=0, value=int(medians['xp']))
headshots = st.number_input("Nombre de headshots", min_value=0, value=int(medians['headshots']))
games_played = st.number_input("Nombre de games played", min_value=0, value=int(medians['games_played']))
time_played = st.number_input("Temps joué", min_value=0, value=int(medians['time_played']))

# Préparation des données pour la prédiction
donnees_utilisateur = np.array([[kills, deaths, losess, xp, headshots, games_played, time_played]])

# Bouton de prédiction et affichage des résultats
if st.button("Prédire"):
    predictions_utilisateur = np.array([tree.predict(donnees_utilisateur) for tree in trees])
    prediction_finale_utilisateur = (np.sum(predictions_utilisateur, axis=0) >= n_estimators / 2).astype(int)

    if prediction_finale_utilisateur[0] == 1:
        st.success("🎉 Prédiction : Le nombre de 'wins' est probablement **au-dessus** de la médiane.")
    else:
        st.error("❌ Prédiction : Le nombre de 'wins' est probablement **en-dessous** de la médiane.")
