import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# Charger les données et sélectionner les colonnes nécessaires
dataTomClancy = pd.read_csv('./rs6_clean.csv')
colonnes = ['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played', 'wins']
GoodDataTomClancy = dataTomClancy[colonnes]

# Charger le modèle d'ensemble d'arbres
trees = load('ensemble_trees.joblib')
n_estimators = len(trees)
print("Le modèle d'ensemble a été chargé avec succès.")

# Calcul des médianes des données (sans 'wins' pour les prédictions)
medians = GoodDataTomClancy[['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played']].median()

# Configurer la page Streamlit
st.set_page_config(page_title="Prédiction Wins", page_icon="🎮", layout="centered", initial_sidebar_state="collapsed")
st.title("🎮 Prédiction des Wins")
st.markdown("### Entrez les caractéristiques du joueur pour prédire si le nombre de wins est supérieur à la médiane")

# Appliquer un style clair et épuré
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

# Afficher les valeurs médianes comme guide
st.sidebar.header("Valeurs médianes pour chaque caractéristique")
for feature, median_value in medians.items():
    st.sidebar.write(f"{feature.capitalize()} : {int(median_value)}")

# Collecter les entrées de l'utilisateur
st.subheader("Caractéristiques")
kills = st.number_input("Nombre de kills", min_value=0, value=int(medians['kills']))
deaths = st.number_input("Nombre de deaths", min_value=0, value=int(medians['deaths']))
losess = st.number_input("Nombre de losess", min_value=0, value=int(medians['losess']))
xp = st.number_input("Nombre de XP", min_value=0, value=int(medians['xp']))
headshots = st.number_input("Nombre de headshots", min_value=0, value=int(medians['headshots']))
games_played = st.number_input("Nombre de games played", min_value=0, value=int(medians['games_played']))
time_played = st.number_input("Temps joué", min_value=0, value=int(medians['time_played']))

# Stocker les entrées de l'utilisateur sous forme de tableau avec les caractéristiques utilisées dans l'entraînement (sans 'wins')
donnees_utilisateur = np.array([[kills, deaths, losess, xp, headshots, games_played, time_played]])

# Bouton pour lancer la prédiction
if st.button("Prédire"):
    # Utiliser le modèle chargé pour prédire
    predictions_utilisateur = [tree.predict(donnees_utilisateur) for tree in trees]
    predictions_utilisateur = np.array(predictions_utilisateur)

    # Calcul du vote majoritaire
    prediction_finale_utilisateur = (np.sum(predictions_utilisateur, axis=0) >= n_estimators / 2).astype(int)

    # Afficher le résultat de la prédiction
    if prediction_finale_utilisateur[0] == 1:
        st.success("🎉 Prédiction : Le nombre de 'wins' est probablement **au-dessus** de la médiane.")
    else:
        st.error("❌ Prédiction : Le nombre de 'wins' est probablement **en-dessous** de la médiane.")
