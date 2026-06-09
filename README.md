# Projet 7 - Systeme de Scoring Credit (MLOps)

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.1-black.svg)
![Streamlit](https://img.shields.io/badge/streamlit-dashboard-red.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-v2-green.svg)
![MLflow](https://img.shields.io/badge/MLflow-tracking-orange.svg)
![SHAP](https://img.shields.io/badge/SHAP-explainability-yellow.svg)
![Evidently](https://img.shields.io/badge/Evidently-data%20drift-lightgrey.svg)

## Presentation

Ce projet OpenClassrooms met en place une chaine MLOps complete pour le credit scoring de la societe "Pret a depenser" :
- modelisation du risque de defaut
- optimisation d'un seuil metier
- deploiement API cloud
- dashboard d'aide a la decision
- suivi du drift

Contrainte metier principale : un Faux Negatif (defaut non detecte) coute 10 fois plus qu'un Faux Positif.

## Resultats cles

- Modele final : LightGBM v2
- AUC : 0.7835
- Cout metier moyen : 0.495
- Recall defaut : 67.0%
- Seuil optimal : 0.50
- Defauts detectes : 3326

## Nouveautes integrees (mise a jour)

- ajout d'un modele lineaire de reference (LogisticRegression) dans le notebook
- ajout d'une interpretabilite locale SHAP (waterfall client)
- ajout des transformations log1p sur variables tres asymetriques
- sauvegarde automatique du seuil dans threshold.json
- sauvegarde des importances globales dans data/feature_importances.csv
- API alignee sur le seuil metier via threshold.json (plus de valeur hardcodee)
- dashboard branche sur de vraies importances et un SHAP local reel
- workflow CI/CD complete avec deploiement Render apres tests
- notebook drift enrichi avec interpretation et conclusion

## Architecture

```text
.
|-- app.py
|-- dashboard.py
|-- model_lgbm.pkl
|-- threshold.json
|-- test_api.py
|-- requirements.txt
|-- data/
|   |-- application_train.csv
|   |-- application_test.csv
|   |-- data_engineered_df.csv
|   |-- feature_importances.csv
|   `-- sample_test.csv
|-- notebooks/
|   |-- Nanecou_Mathilde_2_notebook_modélisation_012026.ipynb
|   |-- analyse_drift.ipynb
|   `-- data_drift_report.html
`-- .github/workflows/testing.yml
```

## Composants

### 1) Notebook de modelisation

Fichier principal : notebooks/Nanecou_Mathilde_2_notebook_modélisation_012026.ipynb

Pipeline realise :
- EDA et preparation
- feature engineering multi-tables (Home Credit)
- baseline Dummy puis LogisticRegression puis LightGBM
- optimisation du seuil selon la fonction de cout FNx10 + FPx1
- interpretabilite SHAP globale et locale
- tracking MLflow et enregistrement du modele final

Sorties generees par le notebook :
- model_lgbm.pkl
- threshold.json
- data/feature_importances.csv
- data/sample_test.csv

### 2) API Flask

Fichier : app.py

Endpoint :
- GET /predict?id=SK_ID_CURR

Reponse :
- probability
- decision (Accorde ou Refuse)
- threshold
- shap_values (top 10 features les plus influentes pour ce client)

L'API charge threshold.json au demarrage. Si le fichier est absent, fallback a 0.5.

API cloud deployee :
- https://api-scoring-mathilde.onrender.com/predict?id=100001

### 3) Dashboard Streamlit

Fichier : dashboard.py

Fonctionnalites :
- appel de l'API cloud
- affichage probabilite et decision
- importance globale (data/feature_importances.csv)
- comparaison client vs population
- explication locale SHAP calculee en temps reel

### 4) Tracking MLflow

Dossier : mlruns/

Permet de retrouver :
- parametres
- metriques
- artefacts (courbes, matrices, graphes SHAP)
- versions de modeles (Registry)

## Tests

Fichier : test_api.py

Couverture principale :
- route racine
- erreurs d'entree (id manquant, id invalide)
- id inconnu
- prediction valide
- verification du seuil metier

Execution :

```bash
pytest test_api.py -v
```

Etat actuel : 10/10 tests passent.

## CI/CD

Workflow : .github/workflows/testing.yml

Sur push/PR :
1. installation des dependances
2. execution des tests Pytest

Sur push main uniquement :
3. deploiement automatique via webhook Render

## Analyse Data Drift

Notebook : notebooks/analyse_drift.ipynb

Livrable HTML : notebooks/data_drift_report.html

Approche : comparaison train/test via Evidently pour detecter les variables en derive.

Conclusion actuelle : drift present mais modere, monitoring mensuel recommande et re-entrainement si derive marquee sur les variables les plus critiques.

## Installation locale

Le projet contient deja un environnement virtuel local p7_env.

### Option A - Utiliser l'environnement existant (Windows)

```powershell
.\p7_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Option B - Creer un nouvel environnement

```powershell
python -m venv p7_env
.\p7_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancer le projet

### 1) API locale

```powershell
python app.py
```

### 2) Dashboard

```powershell
streamlit run dashboard.py
```

### 3) MLflow UI

```powershell
mlflow ui --backend-store-uri mlruns
```

Puis ouvrir : http://127.0.0.1:5000

## Donnees

Source : Kaggle "Home Credit Default Risk"

Fichiers principaux utilises :
- data/application_train.csv
- data/application_test.csv
- data/bureau.csv
- data/bureau_balance.csv
- data/previous_application.csv
- data/installments_payments.csv
- data/POS_CASH_balance.csv
- data/credit_card_balance.csv

## Livrables du projet

- notebooks/Nanecou_Mathilde_2_notebook_modélisation_012026.ipynb
- notebooks/analyse_drift.ipynb
- notebooks/data_drift_report.html
- app.py
- dashboard.py
- test_api.py
- model_lgbm.pkl
- threshold.json
- data/feature_importances.csv
- RAPPORT_PROJET.md
- PROMPT_GAMMA_SOUTENANCE.md
- SCRIPT_SOUTENANCE.md

## Auteur

Mathilde Nanecou
Parcours Data Scientist - OpenClassrooms