import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import seaborn as sns
import streamlit as st

# ===========================================================================
# Configuration de la page
# ===========================================================================
st.set_page_config(
    page_title="Dashboard Scoring Credit - Pret a depenser",
    page_icon="",
    layout="wide",
)

# ===========================================================================
# URL de l'API
# ===========================================================================
API_PREDICT_URL = os.getenv(
    "SCORING_API_PREDICT_URL",
    "https://api-scoring-credit-mathilde.onrender.com/predict",
)
API_SIMULATE_URL = os.getenv(
    "SCORING_API_SIMULATE_URL",
    "https://api-scoring-credit-mathilde.onrender.com/simulate",
)

# ===========================================================================
# Palette accessible (Wong colorblind-safe) -- contrastes WCAG AA
# ===========================================================================
PALETTE = {
    "risk": "#D55E00",       # vermillon (risque)           -- 3.87:1 blanc, 4.89:1 sombre
    "safe": "#0072B2",       # bleu (securite)              -- 5.19:1 blanc, 3.64:1 sombre
    "neutral": "#555555",    # gris fonce                   -- 7.46:1 blanc, 2.58:1 sombre
    "hist_pop": "#3B97BF",   # bleu moyen (population)      -- 3.30:1 blanc, 5.73:1 sombre
    "hist_group": "#BF8400", # ambre fonce (groupe sim.)    -- 3.22:1 blanc, 5.86:1 sombre
    "highlight": "#B56D94",  # rose fonce (client)          -- 3.52:1 blanc, 5.17:1 sombre
    "positive_shap": "#D55E00",
    "negative_shap": "#0072B2",
}

# Taille de police augmentee pour l'accessibilite
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10

# Style CSS pour les descriptions accessibles (lecteurs d'ecran)
st.markdown(
    """
    <style>
    .sr-only {
        position: absolute;
        width: 1px;
        height: 1px;
        padding: 0;
        margin: -1px;
        overflow: hidden;
        clip: rect(0, 0, 0, 0);
        white-space: nowrap;
        border: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def alt_text(description: str) -> None:
    """Insere un texte accessible cache visuellement mais lu par les lecteurs d'ecran."""
    st.markdown(
        f'<p class="sr-only" role="img" aria-label="{description}">{description}</p>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# Dictionnaire de traduction des features techniques -> francais lisible
# ===========================================================================
FEATURE_LABELS = {
    "AMT_INCOME_TOTAL": "Revenu total annuel",
    "AMT_CREDIT": "Montant du credit",
    "AMT_ANNUITY": "Montant de l'annuite",
    "AMT_GOODS_PRICE": "Prix du bien finance",
    "CNT_CHILDREN": "Nombre d'enfants",
    "DAYS_BIRTH": "Age",
    "DAYS_EMPLOYED": "Anciennete emploi",
    "DAYS_REGISTRATION": "Anciennete enregistrement",
    "DAYS_ID_PUBLISH": "Anciennete piece d'identite",
    "EXT_SOURCE_1": "Score externe 1",
    "EXT_SOURCE_2": "Score externe 2",
    "EXT_SOURCE_3": "Score externe 3",
    "PAYMENT_RATE": "Taux de paiement",
    "INSTAL_DPD_MEAN": "Retard moyen paiements",
    "INSTAL_AMT_PAYMENT_SUM": "Somme paiements passes",
    "PREV_APP_CREDIT_PERC_MEAN": "Ratio credit/demande moyen",
    "APPROVED_CNT_PAYMENT_MEAN": "Nb paiements approuves moyen",
    "ACTIVE_DAYS_CREDIT_MEAN": "Anciennete credits actifs moy.",
    "CLOSED_DAYS_CREDIT_MAX": "Duree max credit cloture",
    "REGION_POPULATION_RELATIVE": "Densite population region",
    "HOUR_APPR_PROCESS_START": "Heure de la demande",
    "CODE_GENDER": "Genre",
    "FLAG_OWN_CAR": "Possede une voiture",
    "FLAG_OWN_REALTY": "Possede un bien immobilier",
    "NAME_EDUCATION_TYPE": "Niveau d'education",
    "NAME_FAMILY_STATUS": "Situation familiale",
    "ORGANIZATION_TYPE": "Type d'organisation employeur",
    "OWN_CAR_AGE": "Age du vehicule",
    "TOTALAREA_MODE": "Surface du logement",
}


def human_label(feature_name: str) -> str:
    """Renvoie le label lisible d'une feature, ou le nom technique si inconnu."""
    return FEATURE_LABELS.get(feature_name, feature_name)


# ===========================================================================
# Chargement des donnees
# ===========================================================================
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


# ===========================================================================
# Appels API
# ===========================================================================
def get_predict_payload(client_id: int) -> tuple[dict, str]:
    try:
        response = requests.get(
            API_PREDICT_URL,
            params={"id": str(client_id)},
            timeout=90,
        )
    except requests.RequestException as exc:
        return {}, f"API inaccessible : {exc}"

    if response.status_code != 200:
        try:
            message = response.json().get("error", "Erreur inconnue")
        except ValueError:
            message = response.text
        return {}, f"Erreur API ({response.status_code}) : {message}"

    return response.json(), ""


# ===========================================================================
# Utilitaires
# ===========================================================================
def get_numeric_features(dataframe: pd.DataFrame) -> list[str]:
    numeric_columns = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    return [col for col in numeric_columns if col != "SK_ID_CURR"]


def describe_position(series: pd.Series, value: float) -> str:
    rank_pct = (series <= value).mean() * 100
    return f"{rank_pct:.1f}e percentile"


def format_client_value(field: str, value) -> str:
    """Formate une valeur client pour un affichage lisible."""
    if pd.isna(value):
        return "N/A"
    if field == "DAYS_BIRTH":
        return f"{abs(value) / 365:.0f} ans"
    if field == "DAYS_EMPLOYED":
        return f"{abs(value) / 365:.1f} ans"
    if field in ("DAYS_REGISTRATION", "DAYS_ID_PUBLISH"):
        return f"{abs(value) / 365:.1f} ans"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    return str(value)


# ===========================================================================
# Jauge de score Plotly
# ===========================================================================
def build_score_gauge(probability: float, threshold: float) -> go.Figure:
    """Construit une jauge coloree indiquant la probabilite de defaut."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            number={"suffix": " %", "font": {"size": 36}},
            delta={
                "reference": threshold * 100,
                "increasing": {"color": PALETTE["risk"]},
                "decreasing": {"color": PALETTE["safe"]},
                "suffix": " pts",
            },
            title={"text": "Probabilite de defaut", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 2, "tickfont": {"size": 13}},
                "bar": {"color": PALETTE["neutral"], "thickness": 0.25},
                "bgcolor": "white",
                "steps": [
                    {"range": [0, threshold * 100 * 0.6], "color": "#d4edda"},
                    {"range": [threshold * 100 * 0.6, threshold * 100], "color": "#fff3cd"},
                    {"range": [threshold * 100, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": PALETTE["neutral"], "width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=40, r=40),
    )
    return fig


# ===========================================================================
# Chargement
# ===========================================================================
df = load_data()
feature_importances = load_feature_importances()

# ===========================================================================
# En-tete
# ===========================================================================
st.title("Dashboard de Credit Scoring")
st.caption(
    "Outil d'aide a la decision pour les charges de relation client. "
    "Permet d'expliquer de maniere transparente les decisions d'octroi de credit."
)

if df.empty:
    st.error("Impossible de charger application_test.csv depuis data/ ou ../data/.")
    st.stop()

if "SK_ID_CURR" not in df.columns:
    st.error("La colonne SK_ID_CURR est absente des donnees.")
    st.stop()

available_ids = sorted(df["SK_ID_CURR"].dropna().astype(int).unique().tolist())
numeric_features = get_numeric_features(df)

# ===========================================================================
# Barre laterale
# ===========================================================================
st.sidebar.header("Parametres")
selected_id = st.sidebar.selectbox(
    "Selectionner un client (ID)",
    options=available_ids,
    index=0,
    help="Identifiant unique du dossier client.",
)
comparison_feature = st.sidebar.selectbox(
    "Variable de comparaison",
    options=numeric_features,
    index=numeric_features.index("AMT_INCOME_TOTAL") if "AMT_INCOME_TOTAL" in numeric_features else 0,
    format_func=human_label,
    help="Variable utilisee pour comparer le client a la population.",
)
similarity_window = st.sidebar.slider(
    "Fenetre de similarite (+/- %)",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Pourcentage autour de la valeur client pour definir le groupe de clients similaires.",
)
run_button = st.sidebar.button("Analyser ce client", type="primary")

# ===========================================================================
# Section 1 : Importance globale du modele
# ===========================================================================
st.header("Importance globale du modele")
if not feature_importances.empty and {"Feature", "Importance"}.issubset(feature_importances.columns):
    top_n = st.slider(
        "Nombre de variables globales a afficher",
        min_value=5,
        max_value=20,
        value=10,
    )
    top_global = (
        feature_importances.sort_values("Importance", ascending=False)
        .head(top_n)
        .sort_values("Importance", ascending=True)
    )

    fig_global, ax_global = plt.subplots(figsize=(9, 0.45 * top_n + 1))
    bars = ax_global.barh(
        [human_label(f) for f in top_global["Feature"]],
        top_global["Importance"],
        color=PALETTE["safe"],
    )
    ax_global.set_xlabel("Importance", fontsize=FONT_SIZE_LABEL)
    ax_global.set_title(
        "Top variables les plus influentes dans le modele (global)",
        fontsize=FONT_SIZE_TITLE,
    )
    ax_global.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    st.pyplot(fig_global)
    alt_text(
        f"Graphique a barres horizontales montrant les {top_n} variables les plus importantes du modele. "
        f"Les 3 premieres sont : {', '.join(top_global['Feature'].tail(3).iloc[::-1].tolist())}."
    )
    st.caption(
        "Lecture : plus la barre est longue, plus la variable a d'influence sur les predictions du modele "
        "pour l'ensemble des clients. Cela ne presage pas du sens de l'influence (positif ou negatif)."
    )
else:
    st.info("Fichier feature_importances.csv non trouve ou format inattendu.")

# ===========================================================================
# Attente de l'action utilisateur
# ===========================================================================
if not run_button:
    st.info("Choisissez un client dans la barre laterale puis cliquez sur **Analyser ce client**.")
    st.stop()

client_row = df[df["SK_ID_CURR"].astype(int) == int(selected_id)]
if client_row.empty:
    st.error("Client introuvable dans la base locale.")
    st.stop()

client_data = client_row.iloc[0]

# ===========================================================================
# Appel API
# ===========================================================================
with st.spinner("Recuperation du score via l'API..."):
    payload, api_error = get_predict_payload(int(selected_id))

if api_error:
    st.error(api_error)
    st.stop()

probability = float(payload.get("probability", 0.0))
threshold = float(payload.get("threshold", 0.5))
decision = payload.get("decision", "Inconnue")
distance_to_threshold = probability - threshold

# ===========================================================================
# Section 2 : Score et decision
# ===========================================================================
st.header(f"Analyse du client {selected_id}")

# -- Jauge Plotly --
gauge_col, metrics_col = st.columns([3, 2])

with gauge_col:
    st.plotly_chart(
        build_score_gauge(probability, threshold),
        use_container_width=True,
    )

with metrics_col:
    st.metric("Probabilite de defaut", f"{probability:.2%}")
    st.metric("Seuil metier", f"{threshold:.2%}")
    st.metric("Ecart au seuil", f"{distance_to_threshold:+.2%}")
    st.metric("Decision", decision)

# -- Interpretation textuelle --
if probability >= threshold:
    st.warning(
        f"**Interpretation** : la probabilite de defaut ({probability:.2%}) est superieure "
        f"au seuil metier ({threshold:.2%}). Le dossier est **refuse**. "
        f"L'ecart au seuil est de {abs(distance_to_threshold):.2%} points."
    )
else:
    st.success(
        f"**Interpretation** : la probabilite de defaut ({probability:.2%}) est inferieure "
        f"au seuil metier ({threshold:.2%}). Le dossier est **accepte**. "
        f"La marge de securite est de {abs(distance_to_threshold):.2%} points."
    )

# ===========================================================================
# Section 3 : Informations descriptives du client
# ===========================================================================
st.subheader("Informations descriptives du client")
key_fields = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "PAYMENT_RATE",
]
present_fields = [field for field in key_fields if field in df.columns]

if present_fields:
    n_cols = min(4, len(present_fields))
    metric_columns = st.columns(n_cols)
    for idx, field in enumerate(present_fields):
        value = client_data[field]
        display_value = format_client_value(field, value)
        metric_columns[idx % n_cols].metric(human_label(field), display_value)
else:
    st.info("Aucun champ descriptif standard trouve dans les donnees.")

# ===========================================================================
# Section 4 : Explication locale (SHAP) vs Importance globale
# ===========================================================================
st.subheader("Explication du score : importance locale vs globale")
st.caption(
    "A gauche : les variables qui ont le plus influence le score **de ce client specifique** (SHAP). "
    "A droite : les variables les plus importantes pour le modele **en general**. "
    "Comparer les deux permet de voir si ce client est atypique."
)

shap_values = payload.get("shap_values", [])
shap_col, global_col = st.columns(2)

with shap_col:
    st.markdown("**Importance locale (SHAP) -- ce client**")
    if shap_values:
        shap_df = pd.DataFrame(shap_values)
        if {"feature", "shap_value"}.issubset(shap_df.columns):
            shap_df = shap_df.sort_values("shap_value", key=lambda s: s.abs(), ascending=True)
            fig_shap, ax_shap = plt.subplots(figsize=(7, 4.5))
            bar_colors = [
                PALETTE["positive_shap"] if val > 0 else PALETTE["negative_shap"]
                for val in shap_df["shap_value"]
            ]
            ax_shap.barh(
                [human_label(f) for f in shap_df["feature"]],
                shap_df["shap_value"],
                color=bar_colors,
            )
            ax_shap.axvline(0, color=PALETTE["neutral"], linewidth=1)
            ax_shap.set_title(
                "Variables poussant vers le risque (+) ou la securite (-)",
                fontsize=FONT_SIZE_TITLE,
            )
            ax_shap.set_xlabel("Valeur SHAP", fontsize=FONT_SIZE_LABEL)
            ax_shap.tick_params(labelsize=FONT_SIZE_TICK)
            plt.tight_layout()
            st.pyplot(fig_shap)
            n_pos = sum(1 for v in shap_df["shap_value"] if v > 0)
            n_neg = len(shap_df) - n_pos
            alt_text(
                f"Graphique SHAP pour le client {selected_id}. "
                f"{n_pos} variables poussent vers le risque, {n_neg} vers la securite. "
                f"Variable la plus influente : {human_label(shap_df.iloc[-1]['feature'])}."
            )
            st.caption(
                "Lecture : les barres vermillon (vers la droite) augmentent le risque estime. "
                "Les barres bleues (vers la gauche) le diminuent."
            )
        else:
            st.info("Format SHAP inattendu.")
    else:
        st.info("Aucune explication locale transmise par l'API pour ce client.")

with global_col:
    st.markdown("**Importance globale -- ensemble du modele**")
    if not feature_importances.empty and {"Feature", "Importance"}.issubset(feature_importances.columns):
        n_display = len(shap_df) if shap_values and "feature" in pd.DataFrame(shap_values).columns else 10
        top_global_compare = (
            feature_importances.sort_values("Importance", ascending=False)
            .head(n_display)
            .sort_values("Importance", ascending=True)
        )
        fig_glob2, ax_glob2 = plt.subplots(figsize=(7, 4.5))
        ax_glob2.barh(
            [human_label(f) for f in top_global_compare["Feature"]],
            top_global_compare["Importance"],
            color=PALETTE["safe"],
        )
        ax_glob2.set_title(
            f"Top {n_display} variables globales",
            fontsize=FONT_SIZE_TITLE,
        )
        ax_glob2.set_xlabel("Importance", fontsize=FONT_SIZE_LABEL)
        ax_glob2.tick_params(labelsize=FONT_SIZE_TICK)
        plt.tight_layout()
        st.pyplot(fig_glob2)
        alt_text(
            f"Graphique a barres horizontales des {n_display} variables globales les plus importantes. "
            f"Variable la plus importante : {human_label(top_global_compare['Feature'].iloc[-1])}."
        )
        st.caption(
            "Lecture : importance moyenne de chaque variable sur l'ensemble des predictions. "
            "Cela ne presage pas du sens (positif ou negatif) de l'influence."
        )
    else:
        st.info("Fichier feature_importances.csv non disponible pour la comparaison.")

# ===========================================================================
# Section 5 : Comparaison client vs population / groupe similaire
# ===========================================================================
st.subheader("Comparaison du client a la population")
st.caption(
    f"Distribution de la variable **{human_label(comparison_feature)}** : "
    "population totale (bleu clair) et groupe de clients similaires (ambre)."
)

feature_series = pd.to_numeric(df[comparison_feature], errors="coerce").dropna()
client_value = pd.to_numeric(pd.Series([client_data[comparison_feature]]), errors="coerce").iloc[0]

if pd.isna(client_value) or feature_series.empty:
    st.warning("La variable selectionnee ne contient pas assez de donnees numeriques.")
else:
    # Groupe similaire
    margin = abs(client_value) * (similarity_window / 100) if client_value != 0 else feature_series.std() * 0.5
    group_min = client_value - margin
    group_max = client_value + margin
    similar_mask = pd.to_numeric(df[comparison_feature], errors="coerce").between(
        group_min, group_max, inclusive="both"
    )
    similar_group = df[similar_mask]
    similar_series = pd.to_numeric(similar_group[comparison_feature], errors="coerce").dropna()

    # Metriques
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    stat_col1.metric("Position (population totale)", describe_position(feature_series, client_value))
    stat_col2.metric("Clients similaires", f"{len(similar_group)} clients")
    if not similar_series.empty:
        stat_col3.metric(
            "Position (groupe similaire)",
            describe_position(similar_series, client_value),
        )

    # Histogramme superpose
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))

    # Population totale
    sns.histplot(
        feature_series,
        bins=40,
        color=PALETTE["hist_pop"],
        alpha=0.5,
        label="Population totale",
        ax=ax_comp,
    )
    # Groupe similaire
    if not similar_series.empty and len(similar_series) > 1:
        sns.histplot(
            similar_series,
            bins=20,
            color=PALETTE["hist_group"],
            alpha=0.6,
            label=f"Groupe similaire ({len(similar_group)})",
            ax=ax_comp,
        )

    # Position client
    ax_comp.axvline(
        client_value,
        color=PALETTE["highlight"],
        linestyle="--",
        linewidth=2.5,
        label=f"Client {selected_id}",
    )
    # Mediane population
    ax_comp.axvline(
        feature_series.median(),
        color=PALETTE["neutral"],
        linestyle=":",
        linewidth=2,
        label="Mediane population",
    )

    ax_comp.set_title(
        f"Distribution de {human_label(comparison_feature)}",
        fontsize=FONT_SIZE_TITLE,
    )
    ax_comp.set_xlabel(human_label(comparison_feature), fontsize=FONT_SIZE_LABEL)
    ax_comp.set_ylabel("Nombre de clients", fontsize=FONT_SIZE_LABEL)
    ax_comp.tick_params(labelsize=FONT_SIZE_TICK)
    ax_comp.legend(fontsize=FONT_SIZE_TICK)
    plt.tight_layout()
    st.pyplot(fig_comp)
    alt_text(
        f"Histogramme de la variable {human_label(comparison_feature)}. "
        f"Valeur du client : {format_client_value(comparison_feature, client_value)}. "
        f"Mediane population : {format_client_value(comparison_feature, feature_series.median())}. "
        f"Groupe similaire : {len(similar_group)} clients."
    )
    st.caption(
        f"Lecture : la ligne rose en pointilles marque la valeur du client ({format_client_value(comparison_feature, client_value)}). "
        f"La ligne grise represente la mediane de la population ({format_client_value(comparison_feature, feature_series.median())}). "
        "L'histogramme ambre montre la repartition du groupe de clients similaires."
    )

# ===========================================================================
# Section 6 : Analyse bivariee
# ===========================================================================
st.subheader("Analyse bivariee")
st.caption("Visualisez la relation entre deux variables et situez le client par rapport a la population.")

bivar_cols = st.columns(2)
x_feature = bivar_cols[0].selectbox(
    "Variable X",
    options=numeric_features,
    index=0,
    format_func=human_label,
)
y_feature = bivar_cols[1].selectbox(
    "Variable Y",
    options=numeric_features,
    index=1 if len(numeric_features) > 1 else 0,
    format_func=human_label,
)

if x_feature == y_feature:
    st.info("Selectionnez deux variables differentes pour l'analyse bivariee.")
else:
    bivar_df = df[[x_feature, y_feature, "SK_ID_CURR"]].copy()
    bivar_df[x_feature] = pd.to_numeric(bivar_df[x_feature], errors="coerce")
    bivar_df[y_feature] = pd.to_numeric(bivar_df[y_feature], errors="coerce")
    bivar_df = bivar_df.dropna(subset=[x_feature, y_feature])

    sample_df = bivar_df.sample(min(6000, len(bivar_df)), random_state=42) if not bivar_df.empty else bivar_df
    selected_point = bivar_df[bivar_df["SK_ID_CURR"].astype(int) == int(selected_id)]

    fig_bivar, ax_bivar = plt.subplots(figsize=(10, 5))
    ax_bivar.scatter(
        sample_df[x_feature],
        sample_df[y_feature],
        s=14,
        alpha=0.25,
        c=PALETTE["hist_pop"],
        label="Population",
    )
    if not selected_point.empty:
        ax_bivar.scatter(
            selected_point[x_feature],
            selected_point[y_feature],
            s=120,
            c=PALETTE["highlight"],
            edgecolors="black",
            linewidths=1.5,
            marker="*",
            zorder=5,
            label=f"Client {selected_id}",
        )

    ax_bivar.set_title(
        f"Relation entre {human_label(x_feature)} et {human_label(y_feature)}",
        fontsize=FONT_SIZE_TITLE,
    )
    ax_bivar.set_xlabel(human_label(x_feature), fontsize=FONT_SIZE_LABEL)
    ax_bivar.set_ylabel(human_label(y_feature), fontsize=FONT_SIZE_LABEL)
    ax_bivar.tick_params(labelsize=FONT_SIZE_TICK)
    ax_bivar.legend(fontsize=FONT_SIZE_TICK)
    plt.tight_layout()
    st.pyplot(fig_bivar)
    alt_text(
        f"Nuage de points montrant la relation entre {human_label(x_feature)} et {human_label(y_feature)}. "
        f"{len(sample_df)} clients affiches. Le client {selected_id} est represente par une etoile."
    )
    st.caption(
        f"Lecture : chaque point bleu clair represente un client de la base. "
        f"L'etoile rose represente le client {selected_id}."
    )

# ===========================================================================
# Section 7 : Simulation de scenario (multi-variables)
# ===========================================================================
st.subheader("Simulation de scenario")
st.caption(
    "Modifiez jusqu'a 3 variables du dossier client et recalculez le score via l'API. "
    "Cela permet d'explorer l'impact de changements sur la decision."
)

with st.form("simulation_form"):
    sim_cols = st.columns(3)
    overrides = {}
    for i in range(3):
        with sim_cols[i]:
            feat = st.selectbox(
                f"Variable {i + 1}",
                options=["(aucune)"] + numeric_features,
                index=0,
                format_func=lambda x: human_label(x) if x != "(aucune)" else "(aucune)",
                key=f"sim_feat_{i}",
            )
            if feat != "(aucune)":
                current = float(
                    pd.to_numeric(pd.Series([client_data[feat]]), errors="coerce").fillna(0).iloc[0]
                )
                new_val = st.number_input(
                    f"Nouvelle valeur",
                    value=current,
                    key=f"sim_val_{i}",
                )
                overrides[feat] = new_val

    simulate = st.form_submit_button("Lancer la simulation", type="primary")

if simulate:
    if not overrides:
        st.info("Selectionnez au moins une variable a modifier.")
    else:
        simulation_payload = {
            "id": int(selected_id),
            "overrides": overrides,
        }
        try:
            sim_response = requests.post(API_SIMULATE_URL, json=simulation_payload, timeout=90)
            if sim_response.status_code == 200:
                sim_data = sim_response.json()
                sim_prob = float(sim_data.get("probability", 0.0))
                sim_decision = sim_data.get("decision", "Inconnue")
                delta_prob = sim_prob - probability

                st.success("Simulation effectuee.")

                sim_res_cols = st.columns(4)
                sim_res_cols[0].metric("Probabilite initiale", f"{probability:.2%}")
                sim_res_cols[1].metric("Probabilite simulee", f"{sim_prob:.2%}", delta=f"{delta_prob:+.2%}")
                sim_res_cols[2].metric("Decision initiale", decision)
                sim_res_cols[3].metric("Decision simulee", sim_decision)

                # Resume textuel
                changes_text = ", ".join(
                    [f"**{human_label(k)}** = {v}" for k, v in overrides.items()]
                )
                st.info(
                    f"Variables modifiees : {changes_text}. "
                    f"La probabilite passe de {probability:.2%} a {sim_prob:.2%} "
                    f"(variation de {delta_prob:+.2%} points)."
                )
            else:
                st.warning(
                    "Endpoint /simulate indisponible sur l'API actuelle. "
                    "Deployez la version API mise a jour pour activer cette fonctionnalite."
                )
        except requests.RequestException as exc:
            st.warning(f"Simulation impossible pour le moment : {exc}")

# ===========================================================================
# Section 8 : Note sur l'accessibilite
# ===========================================================================
st.divider()
st.subheader("Accessibilite")
st.markdown(
    """
- **Palette accessible** : couleurs choisies pour etre distinguables par les personnes daltoniennes
  (palette Wong adaptee). Tous les elements graphiques respectent un contraste minimum de 3:1 (WCAG 1.4.11 AA).
- **Double codage** (WCAG 1.4.1) : l'information n'est jamais vehiculee uniquement par la couleur.
  Chaque graphique est accompagne d'un texte de lecture decrivant ce qu'il montre.
- **Contenu non textuel** (WCAG 1.1.1) : chaque graphique dispose d'une description accessible
  invisible a l'ecran mais lue par les lecteurs d'ecran (attribut aria-label).
- **Valeurs explicites** : le score, le seuil et l'ecart sont affiches en texte numerique clair.
- **Titres et structure** (WCAG 2.4.2, 2.4.6) : titre de page descriptif, hierarchie de titres
  coherente (h1/h2/h3), tous les graphiques ont un titre, des labels d'axes et une legende.
- **Redimensionnement** (WCAG 1.4.4) : l'interface est responsive et le texte peut etre agrandi
  a 200%% via le zoom navigateur sans perte de fonctionnalite.
- **Labels en francais** : les noms techniques des variables sont traduits en termes comprehensibles
  pour les personnes non expertes en data science.
"""
)
