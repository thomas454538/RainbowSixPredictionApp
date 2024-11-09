import streamlit as st
import pandas as pd
from joblib import load
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Prédiction des Wins",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 8px 16px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-weight: bold;
            font-size: 1.2em;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        .result-box.positive {
            background-color: rgba(40, 167, 69, 0.9);
            border: 1px solid #28a745;
        }
        .result-box.negative {
            background-color: rgba(220, 53, 69, 0.9);
            border: 1px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_model():
    model_filename = 'random_forest_model.joblib'
    if not os.path.isfile(model_filename):
        st.error(f"Le fichier {model_filename} n'existe pas.")
        st.stop()
    return load(model_filename)

model = load_model()

st.title("🎮 Prédiction des Wins")
st.markdown("### Entrez les caractéristiques du joueur pour prédire les résultats")

for feature in features:
    if feature not in st.session_state:
        st.session_state[feature] = 0

def randomize_inputs():
    random_row = GoodDataTomClancy.sample(n=1)
    for feature in features:
        st.session_state[feature] = int(random_row[feature].values[0])

col1, col2 = st.columns(2)
col1.button("🎲 Randomiser les valeurs", on_click=randomize_inputs)
predict_button = col2.button("📊 Prédire")

for feature in features:
    label = feature_labels.get(feature, feature)
    st.number_input(
        f"**{label}**", 
        min_value=0, 
        key=feature, 
        help='Temps joué en secondes' if feature == 'time_played' else None
    )

user_data = pd.DataFrame([{feature: st.session_state[feature] for feature in features}])

if predict_button:
    user_data = user_data.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(user_data)
    
    result_class = "positive" if prediction[0] == 1 else "negative"
    result_text = "au-dessus" if prediction[0] == 1 else "en-dessous"
    st.markdown(f"""
    <div class="result-box {result_class}">
        Prédiction : Le nombre de 'wins' est probablement <strong>{result_text}</strong> de la médiane.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Comparaison avec la distribution des valeurs")
    user_values = user_data.iloc[0]

    tabs = st.tabs([feature_labels[feature] for feature in features])

    for tab, feature in zip(tabs, features):
        with tab:
            fig, ax = plt.subplots(figsize=(8, 4))
            
            sns.kdeplot(
                GoodDataTomClancy[feature], 
                color='green', 
                fill=True, 
                ax=ax
            )
            
            ax.axvline(user_values[feature], color='red', linestyle='--', linewidth=2, label=user_values[feature])
            
            ax.set_title(feature_labels.get(feature, feature))
            ax.legend()
            
            st.pyplot(fig)
