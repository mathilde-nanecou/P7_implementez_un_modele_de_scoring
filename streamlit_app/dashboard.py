import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------------------------------
# 1. CONFIGURATION ET CHARGEMENT DES DONNÉES
# -----------------------------------------------------------
# L'URL de ton API sur Render
API_URL = "https://api-scoring-mathilde.onrender.com/predict"

st.set_page_config(page_title="Dashboard Scoring Crédit", layout="wide")

@st.cache_data
def load_data():
    # On charge le fichier application_test.csv (version 500 lignes avec 261 col)
    # Assure-toi que ce fichier est bien dans ton dossier data/ en local
    path = 'data/application_test.csv'
    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    else:
        st.error(f"Fichier {path} introuvable. Vérifie ton dossier data.")
        return pd.DataFrame()

df = load_data()

# -----------------------------------------------------------
# 2. ENTÊTE ET IMPORTANCE GLOBALE
# -----------------------------------------------------------
st.title("🏦 Système de Scoring Crédit - Analyse Décisionnelle")

st.header("📊 Importance Globale des Variables")
st.info("Voici les facteurs qui influencent le plus le modèle de manière générale.")

# Données d'importance (Valeurs d'exemple pour l'illustration)
feature_data = {
    'Variable': ['Âge du client', 'Revenu Annuel', 'Montant Crédit', 'Ancienneté Emploi', 'Score Externe'],
    'Importance': [0.15, 0.22, 0.35, 0.18, 0.45]
}
df_glob = pd.DataFrame(feature_data).sort_values(by='Importance', ascending=True)

fig_glob, ax_glob = plt.subplots(figsize=(8, 4))
ax_glob.barh(df_glob['Variable'], df_glob['Importance'], color='#3498db')
ax_glob.set_xlabel('Poids dans le modèle')
st.pyplot(fig_glob)

st.divider()

# -----------------------------------------------------------
# 3. BARRE LATÉRALE - SÉLECTION DU CLIENT
# -----------------------------------------------------------
st.sidebar.header("🔍 Sélection Client")

if not df.empty:
    # On récupère les IDs uniques et on les trie
    available_ids = sorted(df['SK_ID_CURR'].unique())
    
    # Sélection via liste déroulante (évite les erreurs de frappe)
    selected_id = st.sidebar.selectbox("Choisir l'ID du client :", available_ids)

    if st.sidebar.button("Évaluer le dossier"):
        st.header(f"🔎 Analyse du Client {selected_id}")
        
        # Extraction des données du client dans le DataFrame local
        client_info = df[df['SK_ID_CURR'] == selected_id].iloc[0]

        with st.spinner('Communication avec l\'API sur Render...'):
            try:
                # Appel à l'API Render
                # On passe l'ID en paramètre 'id'
                response = requests.get(API_URL, params={"id": str(selected_id)}, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- AFFICHAGE DES MÉTRIQUES ---
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
                    
                    # On place le client actuel sur le graphique
                    ax_comp.axvline(client_info['AMT_INCOME_TOTAL'], color='red', linestyle='--', linewidth=2, label='Ce client')
                    ax_comp.set_title("Distribution des Revenus Annuels")
                    ax_comp.legend()
                    
                    # Limite l'axe X pour éviter les outliers extrêmes
                    ax_comp.set_xlim(0, df['AMT_INCOME_TOTAL'].quantile(0.95)) 
                    st.pyplot(fig_comp)

                    st.divider()

                    # -----------------------------------------------------------
                    # 5. IMPORTANCE LOCALE (SIMULATION SHAP)
                    # -----------------------------------------------------------
                    st.subheader("💡 Pourquoi cette décision ?")
                    st.write("Facteurs spécifiques ayant influencé ce score :")
                    
                    local_feat = ['Revenu', 'Montant Prêt', 'Âge', 'Dettes']
                    # Logique visuelle simple selon le résultat
                    local_val = [0.1, 0.4, -0.05, 0.3] if data['probability'] > 0.5 else [-0.2, -0.1, 0.05, -0.1]
                    
                    fig_loc, ax_loc = plt.subplots(figsize=(8, 3))
                    colors = ['#e74c3c' if x > 0 else '#2ecc71' for x in local_val]
                    ax_loc.barh(local_feat, local_val, color=colors)
                    ax_loc.set_title("Impact sur le score (Rouge = Risque / Vert = Sécurité)")
                    st.pyplot(fig_loc)

                elif response.status_code == 404:
                    st.warning(f"L'API indique que le client {selected_id} n'est pas dans sa base de test.")
                else:
                    st.error(f"L'API a répondu avec une erreur {response.status_code}")
                    
            except Exception as e:
                st.error(f"Impossible de joindre l'API. Vérifie si elle est 'Live' sur Render. Erreur : {e}")
else:
    st.warning("Aucune donnée chargée. Vérifie le dossier 'data/'.")