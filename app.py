import streamlit as st
import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Prédiction des Wins",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load and prepare data
dataTomClancy = pd.read_csv('./rs6_clean.csv')
features = ['kills', 'deaths', 'losess', 'xp', 'headshots', 'games_played', 'time_played']
feature_labels = {
    'kills': 'Nombre de kills',
    'deaths': 'Nombre de deaths',
    'losess': 'Nombre de losses',
    'xp': 'Nombre de XP',
    'headshots': 'Nombre de headshots',
    'games_played': 'Nombre de parties jouées',
    'time_played': 'Temps joué (en secondes)'
}
GoodDataTomClancy = dataTomClancy[features]

# Model loader with caching
@st.cache_resource
def load_model():
    model_filename = 'random_forest_model.joblib'
    if not os.path.isfile(model_filename):
        st.error(f"Le fichier {model_filename} n'existe pas.")
        st.stop()
    return load(model_filename)

model = load_model()

# Main interface
st.title("🎮 Prédiction des Wins")
st.markdown("### Entrez les caractéristiques du joueur pour prédire les résultats")

# Initialize session state
for feature in features:
    if feature not in st.session_state:
        st.session_state[feature] = 0

def randomize_inputs():
    random_row = GoodDataTomClancy.sample(n=1)
    for feature in features:
        st.session_state[feature] = int(random_row[feature].values[0])

# Buttons for randomization and prediction
col1, col2 = st.columns(2)
col1.button("Randomiser les valeurs", on_click=randomize_inputs)
predict_button = col2.button("Prédire")

# Input collection
for i, feature in enumerate(features):
    label = feature_labels.get(feature, feature)
    st.number_input(label, min_value=0, key=feature, help='Temps joué en secondes' if feature == 'time_played' else None)

# Prepare user data for prediction
user_data = pd.DataFrame([{feature: st.session_state[feature] for feature in features}])

# Basic CSS for a clean look
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: Arial, sans-serif;
    }
    /* Tab styling */
    .stTabs [role="tablist"] .tab-label {
        background: #e1e1e1;
        padding: 10px;
        margin: 0 5px;
        border-radius: 5px;
    }
    .stTabs [role="tablist"] .tab-label[data-selected="true"] {
        background: #2575fc;
        color: #ffffff;
        font-weight: bold;
    }
    /* Button styling */
    div[data-testid="stAppViewContainer"] .stButton > button {
        background-color: #2575fc;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Prediction and display
if predict_button:
    user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(user_data)

    result_color = "#28a745" if prediction[0] == 1 else "#dc3545"
    result_text = "au-dessus" if prediction[0] == 1 else "en-dessous"
    st.markdown(f"""
    <div style='background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2>Prédiction : Le nombre de 'wins' est probablement <strong>{result_text}</strong> de la médiane.</h2>
    </div>
    """, unsafe_allow_html=True)

    # Feature distributions in tabs
    st.markdown("### Comparaison avec la distribution des valeurs")
    user_values = user_data.iloc[0]
    tabs = st.tabs([feature_labels[feature] for feature in features])

    for tab, feature in zip(tabs, features):
        with tab:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(GoodDataTomClancy[feature], bins=30, kde=False, color='skyblue', ax=ax)
            ax.axvline(user_values[feature], color='orange', linestyle='--', linewidth=2, label='Votre valeur')
            ax.set_title(feature_labels.get(feature, feature))
            ax.legend()
            st.pyplot(fig)
