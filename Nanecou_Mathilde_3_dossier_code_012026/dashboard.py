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
# API Render
API_URL = "https://api-scoring-mathilde.onrender.com/predict"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

@st.cache_data
def load_data():
    
    path = 'data/application_test.csv'
    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    else:
        st.error(f"Fichier {path} introuvable. Vérifie ton dossier data.")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """Charge le modèle LightGBM depuis le fichier pickle"""
    try:
        with open('model_lgbm.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Modèle non chargé : {e}")
        return None

@st.cache_data
def load_feature_importances():
    """Charge les importances réelles depuis le CSV généré par le notebook"""
    path = 'data/feature_importances.csv'
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

# on charge les importances réelles depuis le modèle
feat_imp = load_feature_importances()
if feat_imp is not None:
    df_glob = feat_imp.head(10).sort_values(by='Importance', ascending=True)
else:
    # fallback si le CSV n'existe pas encore
    st.warning("Fichier feature_importances.csv non trouvé — affichage par défaut.")
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
            placeholder.info("💡 Note : Le serveur Render peut mettre jusqu'à 60s à sortir de veille pour la première requête.")
            
            try:
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

                    # -----------------------------------------------------------
                    # 4. COMPARAISON AVEC LES AUTRES CLIENTS
                    # -----------------------------------------------------------
                    st.subheader("📈 Comparaison avec l'ensemble des clients")
                    st.write("Position du client (ligne rouge) par rapport à la distribution des revenus.")
                    
                    fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
                    sns.histplot(df['AMT_INCOME_TOTAL'], kde=True, ax=ax_comp, color="skyblue")
                    
                    ax_comp.axvline(client_info['AMT_INCOME_TOTAL'], color='red', linestyle='--', linewidth=2, label='Ce client')
                    ax_comp.set_title("Distribution des Revenus Annuels")
                    ax_comp.legend()
                    
                    ax_comp.set_xlim(0, df['AMT_INCOME_TOTAL'].quantile(0.95)) 
                    st.pyplot(fig_comp)

                    st.divider()

                    # -----------------------------------------------------------
                    # 5. IMPORTANCE LOCALE (SHAP réel)
                    # -----------------------------------------------------------
                    st.subheader("💡 Pourquoi cette décision ?")
                    st.write("Facteurs spécifiques ayant influencé ce score :")
                    
                    if model is not None:
                        try:
                            # on prépare les données du client comme l'API le fait
                            client_row = df[df['SK_ID_CURR'] == selected_id]
                            client_data_clean = client_row.select_dtypes(exclude=['object'])
                            expected_features = model.feature_name_
                            client_data_final = client_data_clean.reindex(columns=expected_features, fill_value=0)
                            
                            # calcul SHAP réel pour ce client
                            explainer_local = shap.TreeExplainer(model)
                            shap_vals = explainer_local.shap_values(client_data_final)
                            
                            if isinstance(shap_vals, list):
                                shap_vals_client = shap_vals[1][0]
                            else:
                                shap_vals_client = shap_vals[0]
                            
                            # top 10 features par impact SHAP
                            shap_df = pd.DataFrame({
                                'Feature': expected_features,
                                'SHAP': shap_vals_client
                            }).sort_values(by='SHAP', key=abs, ascending=False).head(10)
                            
                            fig_loc, ax_loc = plt.subplots(figsize=(8, 4))
                            colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in shap_df['SHAP']]
                            ax_loc.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
                            ax_loc.set_title("Impact SHAP réel (Rouge = Risque / Vert = Sécurité)")
                            plt.tight_layout()
                            st.pyplot(fig_loc)
                        except Exception as e:
                            st.warning(f"SHAP non disponible pour ce client : {e}")
                    else:
                        st.info("Modèle non chargé — impossible de calculer les valeurs SHAP locales.")

                elif response.status_code == 404:
                    st.warning(f"L'API indique que le client {selected_id} n'est pas dans sa base de test.")
                else:
                    st.error(f"L'API a répondu avec une erreur {response.status_code}")
                    
            except Exception as e:
                st.error(f"Impossible de joindre l'API. Vérifie si elle est 'Live' sur Render. Erreur : {e}")
else:
    st.warning("Aucune donnée chargée. Vérifie le dossier 'data/'.")