import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import shap
import os

# -----------------------------------------------------------
# 1. CONFIGURATION ET CHARGEMENT DES DONNÉES
# -----------------------------------------------------------
# URL de ton API Render (vérifie qu'elle finit bien par /predict)
API_URL = "https://api-scoring-credit-mathilde.onrender.com/predict"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

@st.cache_data
def load_data():
    # Correction du chemin : le CSV est dans api/data/
    # Note : Vérifie si ton fichier s'appelle 'application_test.csv' ou 'sample_test.csv'
    path = 'api/data/application_test.csv' 
    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    else:
        # Si le fichier au-dessus n'existe pas, on tente le nom suggéré précédemment
        alt_path = 'api/data/sample_test.csv'
        if os.path.exists(alt_path):
            return pd.read_csv(alt_path)
        st.error(f"Fichier de données introuvable dans api/data/. Vérifie ton dossier sur GitHub.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Charge le modèle LightGBM depuis le dossier api/"""
    # Correction du chemin : le modèle est dans le dossier api/
    path = 'api/model_lgbm.pkl'
    try:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            st.error(f"Fichier {path} introuvable sur le serveur.")
            return None
    except Exception as e:
        st.warning(f"Modèle non chargé : {e}")
        return None

@st.cache_data
def load_feature_importances():
    """Charge les importances depuis le dossier api/data/"""
    path = 'api/data/feature_importances.csv'
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return None

df = load_data()
model = load_model()

# -----------------------------------------------------------
# 2. ENTÊTE ET IMPORTANCE GLOBALE
# -----------------------------------------------------------
st.title("🏦 Système de Scoring Crédit - Analyse Décisionnelle")

st.header("📊 Importance Globale des Variables")
st.info("Voici les facteurs qui influencent le plus le modèle de manière générale.")

feat_imp = load_feature_importances()
if feat_imp is not None:
    df_glob = feat_imp.head(10).sort_values(by='Importance', ascending=True)
else:
    st.warning("Données d'importance globale non trouvées — affichage par défaut.")
    df_glob = pd.DataFrame({
        'Feature': ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1', 'PAYMENT_RATE', 'AMT_CREDIT'],
        'Importance': [450, 380, 320, 280, 250]
    }).sort_values(by='Importance', ascending=True)

fig_glob, ax_glob = plt.subplots(figsize=(8, 4))
ax_glob.barh(df_glob['Feature'], df_glob['Importance'], color='#3498db')
ax_glob.set_xlabel('Importance dans le modèle')
st.pyplot(fig_glob)

st.divider()

# -----------------------------------------------------------
# 3. BARRE LATÉRALE - SÉLECTION DU CLIENT
# -----------------------------------------------------------
st.sidebar.header("🔍 Sélection Client")

if not df.empty:
    available_ids = sorted(df['SK_ID_CURR'].unique())
    selected_id = st.sidebar.selectbox("Choisir l'ID du client :", available_ids)

    if st.sidebar.button("Évaluer le dossier"):
        st.header(f"🔎 Analyse du Client {selected_id}")
        
        client_info = df[df['SK_ID_CURR'] == selected_id].iloc[0]

        with st.spinner('⏳ Analyse en cours...'):
            placeholder = st.empty()
            placeholder.info("💡 Note : Le serveur Render peut mettre jusqu'à 60s à sortir de veille.")
            
            try:
                # Appel à l'API Render
                response = requests.get(API_URL, params={"id": str(selected_id)}, timeout=90)
                placeholder.empty()

                if response.status_code == 200:
                    data = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Probabilité de Défaut", f"{data['probability']:.2%}")
                    with col2:
                        decision = data['decision']
                        color = "green" if decision == "Accordé" else "red"
                        st.markdown(f"### Décision : :{color}[{decision}]")
                    with col3:
                        st.metric("Seuil de risque", f"{data['threshold']:.2%}")

                    st.progress(data['probability'])

                    # 4. COMPARAISON
                    st.subheader("📈 Comparaison avec l'ensemble des clients")
                    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
                    sns.histplot(df['AMT_INCOME_TOTAL'], kde=True, ax=ax_comp, color="skyblue")
                    ax_comp.axvline(client_info['AMT_INCOME_TOTAL'], color='red', linestyle='--', label='Ce client')
                    ax_comp.set_xlim(0, df['AMT_INCOME_TOTAL'].quantile(0.95)) 
                    ax_comp.legend()
                    st.pyplot(fig_comp)

                    st.divider()

                    # 5. IMPORTANCE LOCALE (SHAP)
                    st.subheader("💡 Pourquoi cette décision ?")
                    if model is not None:
                        try:
                            client_row = df[df['SK_ID_CURR'] == selected_id]
                            client_data_clean = client_row.select_dtypes(exclude=['object'])
                            expected_features = model.feature_name_
                            client_data_final = client_data_clean.reindex(columns=expected_features, fill_value=0)
                            
                            explainer_local = shap.TreeExplainer(model)
                            shap_vals = explainer_local.shap_values(client_data_final)
                            
                            if isinstance(shap_vals, list):
                                shap_vals_client = shap_vals[1][0]
                            else:
                                shap_vals_client = shap_vals[0]
                            
                            shap_df = pd.DataFrame({
                                'Feature': expected_features,
                                'SHAP': shap_vals_client
                            }).sort_values(by='SHAP', key=abs, ascending=False).head(10)
                            
                            fig_loc, ax_loc = plt.subplots(figsize=(8, 4))
                            colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in shap_df['SHAP']]
                            ax_loc.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
                            ax_loc.set_title("Impact SHAP (Rouge = Risque / Vert = Sécurité)")
                            plt.tight_layout()
                            st.pyplot(fig_loc)
                        except Exception as e:
                            st.warning(f"SHAP non disponible : {e}")
                    else:
                        st.info("Modèle non chargé sur le Dashboard — SHAP indisponible.")

                elif response.status_code == 404:
                    st.warning(f"Client {selected_id} non trouvé par l'API.")
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    
            except Exception as e:
                st.error(f"L'API est injoignable (Render en veille ?). Erreur : {e}")
else:
    st.warning("Aucune donnée chargée. Vérifie ton dossier 'api/data/'.")