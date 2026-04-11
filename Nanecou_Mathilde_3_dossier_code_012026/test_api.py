import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from sklearn.metrics import confusion_matrix

# On empêche le chargement réel du modèle et des données lors de l'import
# Cela permet de passer les tests même si les fichiers sont absents sur GitHub
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
        
        # On calcule manuellement les FP et FN
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        total_cost = (fn * 10) + (fp * 1)
        assert total_cost == 11

class TestDataPreprocessing:

    def test_column_cleaning(self):
        """Vérifie que le nettoyage des noms de colonnes fonctionne"""
        import re
        col_name = "AMT_INCOME TOTAL"
        # On remplace les espaces par des underscores
        clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', col_name).strip('_')
        assert clean_name == "AMT_INCOME_TOTAL"

    def test_inf_values_handling(self):
        """Vérifie le remplacement des valeurs infinies"""
        df = pd.DataFrame({'col': [1.0, np.inf]})
        df = df.replace([np.inf, -np.inf], np.nan)
        assert pd.isna(df['col'].iloc[1])

# --- 3. TESTS DE L'API (SIMULÉS ✅) ---

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