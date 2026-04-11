import pickle
import json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys

# Gestion de l'import SHAP pour éviter les crashs si absent
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = Flask(__name__)
CORS(app)
# =========================================================
# 1. CONFIGURATION DES CHEMINS (Adaptés au dossier /api)
# =========================================================

# On cherche les fichiers là où se trouve app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = 'model_lgbm.pkl'
DATA_PATH = 'data/application_test.csv' 
THRESHOLD_PATH = 'threshold.json'

# =========================================================
# 2. CHARGEMENT DES RESSOURCES
# =========================================================
print("⏳ Démarrage de l'API...")

# A. Chargement du Modèle
model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé")
except Exception as e:
    print(f"❌ Erreur modèle : {e}")

# B. Initialisation SHAP
explainer = None
if model is not None and SHAP_AVAILABLE:
    try:
        # On utilise TreeExplainer pour LightGBM (très rapide)
        explainer = shap.TreeExplainer(model)
        print("✅ Explainer SHAP initialisé")
    except Exception as e:
        print(f"⚠️ SHAP indisponible : {e}")

# C. Chargement des Données (avec sécurité mémoire)
df = None
try:
    # On ne charge que les colonnes nécessaires si possible pour économiser la RAM
    df = pd.read_csv(DATA_PATH)
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    print(f"✅ Données clients chargées ({df.shape[0]} entrées)")
except Exception as e:
    print(f"❌ Erreur données (Vérifiez que le fichier est dans api/data/) : {e}")

# D. Chargement du seuil
OPTIMAL_THRESHOLD = 0.5
try:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            OPTIMAL_THRESHOLD = json.load(f)['threshold']
        print(f"✅ Seuil optimal chargé : {OPTIMAL_THRESHOLD}")
except Exception as e:
    print(f"⚠️ Seuil par défaut utilisé (0.5)")

# =========================================================
# 3. ROUTES
# =========================================================

@app.route('/')
def index():
    return "<h1>API de Scoring Crédit active.</h1><p>Utilisez /predict?id=XXXXXX</p>"

@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.args.get('id')
    
    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400

    if df is None or model is None:
        return jsonify({"error": "Modèle ou données non chargés sur le serveur"}), 500

    try:
        id_int = int(client_id)
        client_row = df[df['SK_ID_CURR'] == id_int]
    except ValueError:
        return jsonify({"error": "ID client doit être un nombre"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client ID {client_id} non trouvé"}), 404

    try:
        # 1. ALIGNEMENT DES FEATURES
        # Récupération des colonnes que le modèle attend
        expected_features = model.feature_name_
        
        # Nettoyage et alignement
        client_data_clean = client_row.select_dtypes(exclude=['object'])
        client_data_final = client_data_clean.reindex(columns=expected_features, fill_value=0)

        # 2. PRÉDICTION
        probability = model.predict_proba(client_data_final)[0][1]
        decision = "Refusé" if probability > OPTIMAL_THRESHOLD else "Accordé"

        # 3. SHAP (Interprétabilité locale)
        shap_top = []
        if explainer is not None:
            try:
                # Calcul des valeurs SHAP pour ce client précis
                shap_vals = explainer.shap_values(client_data_final)
                
                # Gestion du format de sortie SHAP (LightGBM renvoie souvent une liste pour binary class)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0] # On prend la classe 1 (défaut)
                else:
                    sv = shap_vals[0]

                # Extraction du Top 10 des variables influentes
                # On trie par valeur absolue pour avoir l'impact (positif ou négatif)
                indices = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:10]
                
                for i in indices:
                    shap_top.append({
                        "feature": expected_features[i],
                        "shap_value": float(sv[i])
                    })
            except Exception as e:
                print(f"Erreur SHAP : {e}")

        return jsonify({
            "status": "success",
            "client_id": id_int,
            "probability": round(float(probability), 4),
            "decision": decision,
            "threshold": OPTIMAL_THRESHOLD,
            "shap_values": shap_top
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors du calcul : {str(e)}"}), 500

if __name__ == '__main__':
    # Configuration pour Render
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
