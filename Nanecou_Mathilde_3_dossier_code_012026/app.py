import pickle
import json
import pandas as pd
from flask import Flask, jsonify, request
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

app = Flask(__name__)

# Configuration des chemins
MODEL_PATH = 'model_lgbm.pkl'
DATA_PATH = 'data/application_test.csv'
THRESHOLD_PATH = 'threshold.json'

# =========================================================
# 1. CHARGEMENT
# =========================================================
print("⏳ Démarrage de l'API...")

# A. Chargement du Modèle
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Modèle chargé")
except Exception as e:
    print(f"❌ Erreur modèle: {e}")
    model = None

# A-bis. Explainer SHAP (interprétabilité locale)
explainer = None
if model is not None and SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(model)
        print("✅ Explainer SHAP initialisé")
    except Exception as e:
        print(f"⚠️ SHAP indisponible : {e}")

# B. Chargement des Données
try:
    df = pd.read_csv(DATA_PATH)
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    print(f"✅ Données clients chargées ({df.shape[0]} entrées)")
except Exception as e:
    print(f"❌ Erreur données: {e}")
    df = None

# C. Chargement du seuil optimal (calculé dans le notebook)
try:
    with open(THRESHOLD_PATH, 'r') as f:
        OPTIMAL_THRESHOLD = json.load(f)['threshold']
    print(f"✅ Seuil optimal chargé : {OPTIMAL_THRESHOLD}")
except Exception as e:
    OPTIMAL_THRESHOLD = 0.5  # valeur par défaut si le fichier manque
    print(f"⚠️ Seuil par défaut utilisé ({OPTIMAL_THRESHOLD}) : {e}")

# =========================================================
# 2. ROUTES
# =========================================================

@app.route('/')
def index():
    return "<h1>API de Scoring Crédit active.</h1><p>Utilisez /predict?id=XXXXXX</p>"

@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.args.get('id')
    
    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400

    try:
        id_int = int(client_id)
        client_row = df[df['SK_ID_CURR'] == id_int]
    except ValueError:
        return jsonify({"error": "ID client doit être un nombre"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client ID {client_id} non trouvé"}), 404

    try:
        # 1. NETTOYAGE : Suppression des colonnes texte
        client_data_clean = client_row.select_dtypes(exclude=['object'])
        
        # 2. ALIGNEMENT : On force les données à avoir les 261 colonnes du modèle
        # On récupère la liste des colonnes attendues directement depuis le modèle chargé
        expected_features = model.feature_name_
        
        # .reindex() va garder les bonnes colonnes et mettre 0 si une colonne manque
        client_data_final = client_data_clean.reindex(columns=expected_features, fill_value=0)

        # 3. PRÉDICTION : On utilise maintenant le DataFrame filtré à 261 colonnes
        probability = model.predict_proba(client_data_final)[0][1]
        
        threshold = OPTIMAL_THRESHOLD
        decision = "Refusé" if probability > threshold else "Accordé"

        # 4. SHAP : interprétabilité locale (top 10 features)
        shap_top = []
        if explainer is not None:
            try:
                shap_vals = explainer.shap_values(client_data_final)
                # shap_vals peut être une liste [class0, class1] ou un array
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]
                # Top 10 features par valeur absolue
                top_idx = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:10]
                shap_top = [
                    {"feature": expected_features[i], "shap_value": round(float(sv[i]), 6)}
                    for i in top_idx
                ]
            except Exception:
                pass  # SHAP optionnel, ne bloque pas la prédiction

        return jsonify({
            "status": "success",
            "client_id": id_int,
            "probability": float(probability),
            "decision": decision,
            "threshold": threshold,
            "shap_values": shap_top
        })

    except Exception as e:
        # En cas d'erreur, on affiche le message pour comprendre si besoin
        return jsonify({"error": f"Erreur de prédiction : {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)