import pickle
import json
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_lgbm.pkl'
DATA_PATH = 'data/application_test.csv'
THRESHOLD_PATH = 'threshold.json'

model = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Modele charge")
except Exception as e:
    print(f"Erreur chargement modele : {e}")

df = None
try:
    df = pd.read_csv(DATA_PATH)
    if 'TARGET' in df.columns:
        df = df.drop(columns=['TARGET'])
    print(f"Donnees chargees ({df.shape[0]} clients)")
except Exception as e:
    print(f"Erreur chargement donnees : {e}")

OPTIMAL_THRESHOLD = 0.5
try:
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH, 'r') as f:
            OPTIMAL_THRESHOLD = json.load(f)['threshold']
        print(f"Seuil charge : {OPTIMAL_THRESHOLD}")
except Exception as e:
    print(f"Seuil par defaut utilise (0.5) : {e}")


def prepare_client_features(client_frame):
    expected_features = model.feature_name_
    clean = client_frame.select_dtypes(exclude=['object'])
    return clean.reindex(columns=expected_features, fill_value=0)


def get_shap_top(client_data, expected_features):
    shap_top = []
    try:
        contribs = model.booster_.predict(client_data.values, pred_contrib=True)
        sv = contribs[0][:-1]
        indices = sorted(range(len(sv)), key=lambda i: abs(sv[i]), reverse=True)[:10]
        for i in indices:
            shap_top.append({"feature": expected_features[i], "shap_value": float(sv[i])})
    except Exception as e:
        print(f"SHAP indisponible : {e}")
    return shap_top


def build_payload(client_data, client_id, include_shap=True):
    features = model.feature_name_
    prob = model.predict_proba(client_data)[0][1]
    decision = "Refusé" if prob > OPTIMAL_THRESHOLD else "Accordé"
    result = {
        "status": "success",
        "client_id": int(client_id),
        "probability": round(float(prob), 4),
        "decision": decision,
        "threshold": OPTIMAL_THRESHOLD,
    }
    if include_shap:
        result["shap_values"] = get_shap_top(client_data, features)
    return result


@app.route('/')
def index():
    return "<h1>API de Scoring Crédit</h1><p>Utilisez /predict?id=XXXXXX</p>"


@app.route('/predict', methods=['GET'])
def predict():
    client_id = request.args.get('id')

    if not client_id:
        return jsonify({"error": "ID client manquant"}), 400
    if df is None or model is None:
        return jsonify({"error": "Modèle ou données non disponibles"}), 500

    try:
        id_int = int(client_id)
        client_row = df[df['SK_ID_CURR'] == id_int]
    except ValueError:
        return jsonify({"error": "L'ID doit être un nombre entier"}), 400

    if client_row.empty:
        return jsonify({"error": f"Client {client_id} non trouvé"}), 404

    try:
        client_data = prepare_client_features(client_row)
        return jsonify(build_payload(client_data, id_int))
    except Exception as e:
        return jsonify({"error": f"Erreur lors du calcul : {str(e)}"}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    if df is None or model is None:
        return jsonify({"error": "Modèle ou données non disponibles"}), 500

    body = request.get_json(silent=True) or {}
    client_id = body.get('id')
    overrides = body.get('overrides', {})

    if client_id is None:
        return jsonify({"error": "Champ 'id' manquant"}), 400

    try:
        id_int = int(client_id)
    except (ValueError, TypeError):
        return jsonify({"error": "L'ID doit être un nombre entier"}), 400

    if not isinstance(overrides, dict):
        return jsonify({"error": "Le champ 'overrides' doit être un objet JSON"}), 400

    client_row = df[df['SK_ID_CURR'] == id_int]
    if client_row.empty:
        return jsonify({"error": f"Client {client_id} non trouvé"}), 404

    try:
        row = client_row.copy()
        for feature, value in overrides.items():
            if feature not in row.columns:
                return jsonify({"error": f"Variable inconnue : {feature}"}), 400
            row.loc[:, feature] = value

        client_data = prepare_client_features(row)
        result = build_payload(client_data, id_int, include_shap=False)
        result["overrides_applied"] = overrides
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la simulation : {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
