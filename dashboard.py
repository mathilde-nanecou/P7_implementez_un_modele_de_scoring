import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Dashboard Scoring Credit", layout="wide")

API_PREDICT_URL = os.getenv(
    "SCORING_API_PREDICT_URL",
    "https://api-scoring-mathilde.onrender.com/predict",
)
API_SIMULATE_URL = os.getenv(
    "SCORING_API_SIMULATE_URL",
    "https://api-scoring-mathilde.onrender.com/simulate",
)

PALETTE = {
    "risk": "#B00020",
    "safe": "#005A9C",
    "neutral": "#4C4C4C",
    "hist": "#0072B2",
    "highlight": "#E69F00",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    data_paths = [
        "data/application_test.csv",
        "../data/application_test.csv",
    ]
    for path in data_paths:
        if os.path.exists(path):
            data = pd.read_csv(path)
            if "TARGET" in data.columns:
                data = data.drop(columns=["TARGET"])
            return data
    return pd.DataFrame()


@st.cache_data
def load_feature_importances() -> pd.DataFrame:
    importance_paths = [
        "data/feature_importances.csv",
        "../data/feature_importances.csv",
    ]
    for path in importance_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return pd.DataFrame()


def get_predict_payload(client_id: int) -> tuple[dict, str]:
    try:
        response = requests.get(
            API_PREDICT_URL,
            params={"id": str(client_id)},
            timeout=90,
        )
    except requests.RequestException as exc:
        return {}, f"API inaccessible: {exc}"

    if response.status_code != 200:
        try:
            message = response.json().get("error", "Erreur inconnue")
        except ValueError:
            message = response.text
        return {}, f"Erreur API ({response.status_code}): {message}"

    return response.json(), ""


def get_numeric_features(dataframe: pd.DataFrame) -> list[str]:
    numeric_columns = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_columns if col != "SK_ID_CURR"]


def describe_position(series: pd.Series, value: float) -> str:
    rank_pct = (series <= value).mean() * 100
    return f"{rank_pct:.1f}e percentile"


df = load_data()
feature_importances = load_feature_importances()

st.title("Dashboard de Credit Scoring")
st.caption(
    "Outil d'aide a la decision pour expliquer un score de credit de maniere lisible et transparente."
)

if df.empty:
    st.error("Impossible de charger application_test.csv depuis streamlit_app/data ou data.")
    st.stop()

if "SK_ID_CURR" not in df.columns:
    st.error("La colonne SK_ID_CURR est absente des donnees.")
    st.stop()

available_ids = sorted(df["SK_ID_CURR"].dropna().astype(int).unique().tolist())
numeric_features = get_numeric_features(df)

st.sidebar.header("Parametres")
selected_id = st.sidebar.selectbox("Selectionner un client", options=available_ids, index=0)
comparison_feature = st.sidebar.selectbox(
    "Variable principale de comparaison",
    options=numeric_features,
    index=numeric_features.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in numeric_features else 0,
)
similarity_window = st.sidebar.slider(
    "Fenetre de similarite (+/- % autour de la valeur client)",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
)
run_button = st.sidebar.button("Analyser ce client", type="primary")

st.header("Importance globale du modele")
if not feature_importances.empty and {"Feature", "Importance"}.issubset(feature_importances.columns):
    top_n = st.slider("Nombre de variables globales a afficher", min_value=5, max_value=20, value=10)
    top_global = (
        feature_importances.sort_values("Importance", ascending=False)
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )

    fig_global, ax_global = plt.subplots(figsize=(8, 4.5))
    ax_global.barh(top_global["Feature"], top_global["Importance"], color=PALETTE["hist"])
    ax_global.set_xlabel("Importance")
    ax_global.set_ylabel("Variables")
    ax_global.set_title("Top variables influentes (global)")
    st.pyplot(fig_global)
    st.caption("Lecture: plus la barre est longue, plus la variable influence le score global du modele.")
else:
    st.info("Fichier feature_importances.csv non trouve ou format inattendu.")

if not run_button:
    st.info("Choisissez un client dans la barre laterale puis cliquez sur 'Analyser ce client'.")
    st.stop()

client_row = df[df["SK_ID_CURR"].astype(int) == int(selected_id)]
if client_row.empty:
    st.error("Client introuvable dans la base locale.")
    st.stop()

client_data = client_row.iloc[0]

with st.spinner("Recuperation du score via l'API..."):
    payload, api_error = get_predict_payload(int(selected_id))

if api_error:
    st.error(api_error)
    st.stop()

probability = float(payload.get("probability", 0.0))
threshold = float(payload.get("threshold", 0.5))
decision = payload.get("decision", "Inconnue")
distance_to_threshold = probability - threshold

st.header(f"Analyse client: {selected_id}")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Probabilite de defaut", f"{probability:.2%}")
col_b.metric("Seuil metier", f"{threshold:.2%}")
col_c.metric("Ecart au seuil", f"{distance_to_threshold:+.2%}")
col_d.metric("Decision", decision)

st.progress(min(max(probability, 0.0), 1.0))

if probability >= threshold:
    st.warning(
        "Interpretation: le dossier est au-dessus du seuil de risque. La decision actuelle est le refus du credit."
    )
else:
    st.success(
        "Interpretation: le dossier est en dessous du seuil de risque. La decision actuelle est l'accord du credit."
    )

st.subheader("Informations descriptives du client")
key_fields = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "CNT_CHILDREN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "PAYMENT_RATE",
]
present_fields = [field for field in key_fields if field in df.columns]

if present_fields:
    metric_columns = st.columns(min(4, len(present_fields)))
    for idx, field in enumerate(present_fields):
        value = client_data[field]
        display_value = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value)
        if field == "DAYS_BIRTH" and pd.notna(value):
            display_value = f"{abs(value) / 365:.1f} ans"
        if field == "DAYS_EMPLOYED" and pd.notna(value):
            display_value = f"{abs(value) / 365:.1f} ans"
        metric_columns[idx % len(metric_columns)].metric(field, display_value)
else:
    st.info("Aucun champ descriptif standard trouve dans les donnees.")

st.subheader("Comparaison client vs population")
feature_series = pd.to_numeric(df[comparison_feature], errors="coerce").dropna()
client_value = pd.to_numeric(pd.Series([client_data[comparison_feature]]), errors="coerce").iloc[0]

if pd.isna(client_value) or feature_series.empty:
    st.warning("La variable de comparaison selectionnee ne contient pas assez de donnees numeriques.")
else:
    margin = abs(client_value) * (similarity_window / 100)
    group_min = client_value - margin
    group_max = client_value + margin
    similar_group = df[
        pd.to_numeric(df[comparison_feature], errors="coerce").between(group_min, group_max, inclusive="both")
    ]

    stat_col1, stat_col2 = st.columns(2)
    stat_col1.metric("Position du client", describe_position(feature_series, client_value))
    stat_col2.metric("Clients similaires (taille)", f"{len(similar_group)}")

    fig_comp, ax_comp = plt.subplots(figsize=(10, 4.5))
    sns.histplot(feature_series, bins=40, color=PALETTE["hist"], alpha=0.75, ax=ax_comp)
    ax_comp.axvline(client_value, color=PALETTE["highlight"], linestyle="--", linewidth=2, label="Client")
    ax_comp.axvline(feature_series.median(), color=PALETTE["neutral"], linestyle=":", linewidth=2, label="Mediane")
    ax_comp.set_title(f"Distribution de {comparison_feature}")
    ax_comp.set_xlabel(comparison_feature)
    ax_comp.set_ylabel("Nombre de clients")
    ax_comp.legend()
    st.pyplot(fig_comp)
    st.caption(
        "Lecture: la ligne orange montre la valeur du client; la ligne grise represente la mediane de la population."
    )

st.subheader("Analyse bivariee")
bivar_cols = st.columns(2)
x_feature = bivar_cols[0].selectbox("Variable X", options=numeric_features, index=0)
y_feature = bivar_cols[1].selectbox(
    "Variable Y",
    options=numeric_features,
    index=1 if len(numeric_features) > 1 else 0,
)

if x_feature == y_feature:
    st.info("Selectionnez deux variables differentes pour l'analyse bivariee.")
else:
    bivar_df = df[[x_feature, y_feature, "SK_ID_CURR"]].copy()
    bivar_df[x_feature] = pd.to_numeric(bivar_df[x_feature], errors="coerce")
    bivar_df[y_feature] = pd.to_numeric(bivar_df[y_feature], errors="coerce")
    bivar_df = bivar_df.dropna(subset=[x_feature, y_feature])

    # Limite l'affichage pour garder l'interface fluide avec de grands volumes.
    sample_df = bivar_df.sample(min(6000, len(bivar_df)), random_state=42) if not bivar_df.empty else bivar_df
    selected_point = bivar_df[bivar_df["SK_ID_CURR"].astype(int) == int(selected_id)]

    fig_bivar, ax_bivar = plt.subplots(figsize=(10, 5))
    ax_bivar.scatter(
        sample_df[x_feature],
        sample_df[y_feature],
        s=14,
        alpha=0.25,
        c=PALETTE["hist"],
        label="Population",
    )
    if not selected_point.empty:
        ax_bivar.scatter(
            selected_point[x_feature],
            selected_point[y_feature],
            s=90,
            c=PALETTE["highlight"],
            edgecolors="black",
            linewidths=1,
            marker="*",
            label="Client",
        )

    ax_bivar.set_title(f"Relation entre {x_feature} et {y_feature}")
    ax_bivar.set_xlabel(x_feature)
    ax_bivar.set_ylabel(y_feature)
    ax_bivar.legend()
    st.pyplot(fig_bivar)

st.subheader("Explication locale (SHAP)")
shap_values = payload.get("shap_values", [])
if shap_values:
    shap_df = pd.DataFrame(shap_values)
    if {"feature", "shap_value"}.issubset(shap_df.columns):
        shap_df = shap_df.sort_values("shap_value", key=lambda s: s.abs(), ascending=True)
        fig_shap, ax_shap = plt.subplots(figsize=(8.5, 4.5))
        bar_colors = [PALETTE["risk"] if val > 0 else PALETTE["safe"] for val in shap_df["shap_value"]]
        ax_shap.barh(shap_df["feature"], shap_df["shap_value"], color=bar_colors)
        ax_shap.axvline(0, color=PALETTE["neutral"], linewidth=1)
        ax_shap.set_title("Variables qui poussent vers le risque (+) ou vers la securite (-)")
        ax_shap.set_xlabel("Valeur SHAP")
        st.pyplot(fig_shap)
        st.caption("Lecture: + augmente le risque estime, - le diminue.")
    else:
        st.info("Le format des valeurs SHAP recues ne permet pas l'affichage.")
else:
    st.info("Aucune explication locale transmise par l'API pour ce client.")

st.subheader("Simulation de scenario (optionnel)")
st.caption("Permet de tester une modification de variables et de recalculer un score via l'endpoint /simulate.")

sim_columns = st.columns(3)
sim_feature = sim_columns[0].selectbox("Variable a modifier", options=numeric_features)
current_value = float(pd.to_numeric(pd.Series([client_data[sim_feature]]), errors="coerce").fillna(0).iloc[0])
sim_value = sim_columns[1].number_input("Nouvelle valeur", value=current_value)
simulate = sim_columns[2].button("Lancer simulation")

if simulate:
    simulation_payload = {
        "id": int(selected_id),
        "overrides": {sim_feature: sim_value},
    }
    try:
        sim_response = requests.post(API_SIMULATE_URL, json=simulation_payload, timeout=90)
        if sim_response.status_code == 200:
            sim_data = sim_response.json()
            st.success("Simulation effectuee.")
            sim_col1, sim_col2 = st.columns(2)
            sim_col1.metric("Probabilite initiale", f"{probability:.2%}")
            sim_col2.metric("Probabilite simulee", f"{float(sim_data.get('probability', 0.0)):.2%}")
        else:
            st.warning(
                "Endpoint /simulate indisponible sur l'API actuelle. "
                "Deployez la version API mise a jour pour activer cette fonctionnalite."
            )
    except requests.RequestException as exc:
        st.warning(f"Simulation impossible pour le moment: {exc}")

st.subheader("Accessibilite")
st.markdown(
    """
    - Couleurs a contraste renforce (rouge/bleu/ambre) pour limiter les ambiguitees.
    - Chaque graphique est accompagne d'un texte de lecture pour eviter une interpretation basee uniquement sur la couleur.
    - Les valeurs cles (score, seuil, ecart) sont affichees en texte numerique explicite.
    - Les titres et axes de graphiques sont systematiquement renseignes.
    """
)