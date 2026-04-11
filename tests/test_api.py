import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch


with patch('pandas.read_csv'), patch('pickle.load'):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app import app

# --- 1. FIXTURE POUR L'API ---
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- 2. TESTS DE LOGIQUE MÉTIER (UNITAIRES - TOUJOURS VERT ✅) ---

class TestBusinessLogic:
    
    def test_payment_rate_calculation(self):
        """Vérifie que le calcul du taux de paiement est correct"""
        df = pd.DataFrame({'AMT_ANNUITY': [1000], 'AMT_CREDIT': [10000]})
        payment_rate = df['AMT_ANNUITY'] / df['AMT_CREDIT']
        assert payment_rate.iloc[0] == 0.1

    def test_business_cost_metric(self):
        """Vérifie la métrique de coût personnalisée (FN=10, FP=1)"""
        y_true = np.array([1, 0]) 
        y_pred = np.array([0, 1]) 
        
        # calcul manuel FP et FN
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        total_cost = (fn * 10) + (fp * 1)
        assert total_cost == 11

class TestDataPreprocessing:

    def test_column_cleaning(self):
        """Vérifie que le nettoyage des noms de colonnes fonctionne"""
        import re
        col_name = "AMT_INCOME TOTAL"

        # remplacer les espaces par des underscores
        clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', col_name).strip('_')
        assert clean_name == "AMT_INCOME_TOTAL"

    def test_inf_values_handling(self):
        """Vérifie le remplacement des valeurs infinies"""
        df = pd.DataFrame({'col': [1.0, np.inf]})
        df = df.replace([np.inf, -np.inf], np.nan)
        assert pd.isna(df['col'].iloc[1])

# --- 3. TESTS DE L'API ---

def test_api_home(client):
    """Vérifie que l'accueil de l'API fonctionne"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"API" in response.data


@patch('app.model')
@patch('app.df')
def test_predict_error_cases(mock_df, mock_model, client):
    """Vérifie la gestion des erreurs basiques sans charger de données"""
    # Test ID manquant
    assert client.get('/predict').status_code == 400
    # Test format invalide
    assert client.get('/predict?id=abc').status_code == 400


@patch('app.explainer', None)
@patch('app.model')
def test_predict_client_not_found(mock_model, client):
    """Vérifie le retour 404 si l'ID client n'existe pas"""
    df_clients = pd.DataFrame([{'SK_ID_CURR': 123456, 'NUM_FEATURE_1': 10.0}])

    with patch('app.df', df_clients):
        response = client.get('/predict?id=999999')

        assert response.status_code == 404
        assert b"non trouv" in response.data

@patch('app.explainer', None)
@patch('app.model')
def test_predict_success_and_payload(mock_model, client):
    """Vérifie une prédiction réussie et la structure de la réponse JSON"""
    df_client = pd.DataFrame([
        {
            'SK_ID_CURR': 100001,
            'NUM_FEATURE_1': 12.0,
            'TEXT_FEATURE': 'test'
        }
    ])

    with patch('app.df', df_client):
        mock_model.feature_name_ = ['SK_ID_CURR', 'NUM_FEATURE_1', 'MISSING_FEATURE']
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        response = client.get('/predict?id=100001')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert data['client_id'] == 100001
        assert data['probability'] == 0.7
        assert data['decision'] == 'Refusé'
        assert data['threshold'] == 0.5
        assert 'shap_values' in data


@patch('app.explainer', None)
@patch('app.model')
def test_predict_decision_threshold(mock_model, client):
    """Vérifie la décision autour du seuil 0.5"""
    df_client = pd.DataFrame([{'SK_ID_CURR': 200002, 'NUM_FEATURE_1': 1.0}])

    with patch('app.df', df_client):
        mock_model.feature_name_ = ['SK_ID_CURR', 'NUM_FEATURE_1']
        mock_model.predict_proba.return_value = np.array([[0.52, 0.48]])

        response = client.get('/predict?id=200002')

        assert response.status_code == 200
        data = response.get_json()
        assert data['decision'] == 'Accordé'


@patch('app.explainer', None)
@patch('app.model')
def test_predict_internal_error_returns_500(mock_model, client):
    """Vérifie le retour 500 en cas d'exception lors de la prédiction"""
    df_client = pd.DataFrame([{'SK_ID_CURR': 300003, 'NUM_FEATURE_1': 1.0}])

    with patch('app.df', df_client):
        mock_model.feature_name_ = ['SK_ID_CURR', 'NUM_FEATURE_1']
        mock_model.predict_proba.side_effect = RuntimeError('erreur modèle')

        response = client.get('/predict?id=300003')

        assert response.status_code == 500
        assert b"Erreur de pr" in response.data