"""
PROJET H2 — Prédiction de la potabilité de l'eau
ÉTAPE 4 : Application Streamlit — Outil terrain
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# CONFIG & CHARGEMENT
# ─────────────────────────────────────────
st.set_page_config(
    page_title="💧 Potabilité de l'eau",
    page_icon="💧",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("best_model_name.txt") as f:
        model_name = f.read().strip()
    threshold = float(np.load("optimal_threshold.npy"))
    return model, scaler, model_name, threshold

model, scaler, model_name, threshold = load_artifacts()

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines",
    "Sulfate", "Conductivity", "Organic_carbon",
    "Trihalomethanes", "Turbidity"
]

WHO_NORMS = {
    "ph":               (6.5,  8.5),
    "Hardness":         (0,    300),
    "Solids":           (0,    500),
    "Chloramines":      (0,    4.0),
    "Sulfate":          (0,    250),
    "Conductivity":     (0,    400),
    "Organic_carbon":   (0,    2.0),
    "Trihalomethanes":  (0,    80),
    "Turbidity":        (0,    4.0),
}

DESCRIPTIONS = {
    "ph":               "Acidité / Alcalinité (6.5 – 8.5 OMS)",
    "Hardness":         "Dureté de l'eau en mg/L",
    "Solids":           "Solides dissous totaux en ppm",
    "Chloramines":      "Chloramines en ppm (max 4 OMS)",
    "Sulfate":          "Sulfates en mg/L (max 250 OMS)",
    "Conductivity":     "Conductivité électrique en μS/cm",
    "Organic_carbon":   "Carbone organique en ppm (max 2 OMS)",
    "Trihalomethanes":  "Trihalométhanes en μg/L (max 80 OMS)",
    "Turbidity":        "Turbidité en NTU (max 4 OMS)",
}

TYPICAL_VALUES = {
    "ph": 7.08, "Hardness": 196.4, "Solids": 20927.0,
    "Chloramines": 7.12, "Sulfate": 333.8, "Conductivity": 426.2,
    "Organic_carbon": 14.3, "Trihalomethanes": 66.4, "Turbidity": 3.99
}

SLIDER_RANGES = {
    "ph":               (0.0,   14.0,  0.01),
    "Hardness":         (0.0,   500.0, 0.1),
    "Solids":           (0.0,   60000.0, 1.0),
    "Chloramines":      (0.0,   15.0,  0.01),
    "Sulfate":          (0.0,   600.0, 0.1),
    "Conductivity":     (0.0,   800.0, 0.1),
    "Organic_carbon":   (0.0,   30.0,  0.01),
    "Trihalomethanes":  (0.0,   130.0, 0.1),
    "Turbidity":        (0.0,   10.0,  0.01),
}

# ─────────────────────────────────────────
# MODÈLES QUI NÉCESSITENT LE SCALING
# ─────────────────────────────────────────
# ✅ FIX 1 : liste étendue — si le meilleur modèle change (SVM, LR, Ensemble Voting),
# le scaling est appliqué correctement.
MODELS_NEEDING_SCALING = ["Logistic Regression", "SVM", "Ensemble Voting"]

# ─────────────────────────────────────────
# HELPER SHAP — compatibilité toutes versions
# ─────────────────────────────────────────
def extract_shap_values_class1(shap_output):
    """
    ✅ FIX 2 : extrait les valeurs SHAP pour la classe 1 (potable),
    quelle que soit la version de SHAP :
    - Ancienne API : liste [classe0, classe1] → shape (n, 9) chacun
    - Nouvelle API : array 3D              → shape (n, 9, 2)
    - XGBoost récent : array 2D            → shape (n, 9)
    """
    if isinstance(shap_output, list):
        # Ancienne API : liste de 2 arrays
        return shap_output[1]
    arr = np.array(shap_output)
    if arr.ndim == 3:
        # Nouvelle API 3D : (n_samples, n_features, n_classes)
        return arr[:, :, 1]
    # Array 2D : déjà les valeurs pour la classe cible
    return arr

# ─────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────
st.title("💧 Prédiction de la Potabilité de l'Eau")
st.markdown(f"""
*Outil terrain — Afrique de l'Ouest* &nbsp;|&nbsp; 
Modèle : **{model_name}** &nbsp;|&nbsp; 
Seuil de décision : **{threshold:.2f}**
""")

st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — SAISIE DES PARAMÈTRES
# ─────────────────────────────────────────
st.sidebar.header("🔬 Paramètres de l'échantillon")
st.sidebar.markdown("Saisissez les valeurs mesurées sur le terrain :")

input_values = {}
for feat in FEATURES:
    mn, mx, step = SLIDER_RANGES[feat]
    val = st.sidebar.number_input(
        label=feat,
        min_value=float(mn),
        max_value=float(mx),
        value=float(TYPICAL_VALUES[feat]),
        step=float(step),
        help=DESCRIPTIONS[feat]
    )
    input_values[feat] = val

predict_btn = st.sidebar.button("🔍 Analyser l'échantillon", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────
if predict_btn:
    X_input = pd.DataFrame([input_values])
    needs_scaling = model_name in MODELS_NEEDING_SCALING

    # ✅ FIX 3 : toujours passer un DataFrame avec noms de colonnes au modèle
    # Évite le warning "X does not have valid feature names"
    if needs_scaling:
        X_scaled = pd.DataFrame(scaler.transform(X_input), columns=FEATURES)
    else:
        X_scaled = X_input.copy()

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= threshold)

    # ── Résultat principal ──
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        if prediction == 1:
            st.success("# ✅ EAU POTABLE")
            st.metric("Probabilité de potabilité", f"{prob:.1%}")
        else:
            st.error("# ❌ EAU NON POTABLE")
            st.metric("Probabilité de potabilité", f"{prob:.1%}")

        st.progress(prob, text=f"Score : {prob:.3f} (seuil = {threshold:.2f})")

    st.markdown("---")

    # ── Paramètres hors norme ──
    st.subheader("📋 Analyse des paramètres")

    col_a, col_b = st.columns(2)
    out_of_range = []
    ok_params = []

    for feat, val in input_values.items():
        lo, hi = WHO_NORMS[feat]
        if val < lo or val > hi:
            out_of_range.append((feat, val, lo, hi))
        else:
            ok_params.append((feat, val))

    with col_a:
        if out_of_range:
            st.markdown("**🔴 Paramètres hors norme OMS :**")
            for feat, val, lo, hi in out_of_range:
                st.error(f"**{feat}** = {val:.3f}  _(norme : {lo} – {hi})_")
        else:
            st.success("Tous les paramètres sont dans les normes OMS.")

    with col_b:
        st.markdown("**🟢 Paramètres conformes :**")
        for feat, val in ok_params:
            lo, hi = WHO_NORMS[feat]
            st.success(f"{feat} = {val:.3f}  _(norme : {lo} – {hi})_")

    st.markdown("---")

    # ── SHAP local ──
    st.subheader("🧠 Explication de la décision (SHAP)")

    try:
        if model_name in ["Random Forest", "XGBoost", "Gradient Boosting"]:
            explainer = shap.TreeExplainer(model)
            shap_output = explainer.shap_values(X_scaled)
            # ✅ FIX 2 appliqué : extraction robuste quelle que soit la version SHAP
            shap_matrix = extract_shap_values_class1(shap_output)
            sv = shap_matrix[0]  # 1 seul échantillon → on prend la première ligne
        else:
            df_bg = pd.read_csv("water_potability_clean.csv").drop("Potability", axis=1)
            if needs_scaling:
                X_bg = pd.DataFrame(scaler.transform(df_bg), columns=FEATURES)
            else:
                X_bg = df_bg.copy()
            explainer = shap.KernelExplainer(model.predict_proba, X_bg[:100])
            shap_output = explainer.shap_values(X_scaled)
            shap_matrix = extract_shap_values_class1(shap_output)
            sv = shap_matrix[0]

        # Graphique waterfall
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["tomato" if v > 0 else "steelblue" for v in sv]
        bars = ax.barh(FEATURES, sv, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Contribution SHAP (+ = vers potable)")
        ax.set_title("Contribution de chaque paramètre à la décision")
        for bar, val in zip(bars, sv):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8
            )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    except Exception as e:
        st.warning(f"SHAP non disponible pour cet échantillon : {e}")

    # ── Tableau récapitulatif ──
    st.markdown("---")
    st.subheader("📊 Tableau des valeurs saisies")
    recap = pd.DataFrame({
        "Paramètre": FEATURES,
        "Valeur": [input_values[f] for f in FEATURES],
        "Norme OMS min": [WHO_NORMS[f][0] for f in FEATURES],
        "Norme OMS max": [WHO_NORMS[f][1] for f in FEATURES],
        "Statut": [
            "🔴 Hors norme" if (input_values[f] < WHO_NORMS[f][0] or input_values[f] > WHO_NORMS[f][1])
            else "🟢 Conforme"
            for f in FEATURES
        ]
    })
    st.dataframe(recap, use_container_width=True)

else:
    # ── Écran d'accueil ──
    st.info("👈 Saisissez les paramètres dans le panneau gauche, puis cliquez sur **Analyser l'échantillon**.")

    st.markdown("### Comment utiliser cet outil ?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Mesurer**")
        st.markdown("Prélevez un échantillon d'eau et mesurez les 9 paramètres physico-chimiques avec vos équipements de terrain.")
    with col2:
        st.markdown("**2️⃣ Saisir**")
        st.markdown("Entrez les valeurs dans le panneau de gauche. Les valeurs pré-remplies sont des valeurs typiques du dataset.")
    with col3:
        st.markdown("**3️⃣ Analyser**")
        st.markdown("Cliquez sur **Analyser** — l'outil indique si l'eau est potable, les paramètres hors norme, et explique la décision.")

    st.markdown("---")
    st.markdown(f"""
    **Informations techniques**  
    - Modèle : `{model_name}`  
    - Seuil de décision optimisé : `{threshold:.2f}` (calibré pour minimiser les faux négatifs dangereux)  
    - Dataset d'entraînement : Water Potability Dataset — 3 276 échantillons (Kaggle)  
    - Critères OMS utilisés pour la détection des paramètres hors norme
    """)