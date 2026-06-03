"""
PROJET H2 — Prédiction de la potabilité de l'eau
ÉTAPE 4 : Application Streamlit — Outil terrain (Version corrigée)
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

# Normes OMS pour chaque paramètre [min_ok, max_ok]
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

# VALEURS PLAUSIBLES pour validation
PLAUSIBLE_RANGES = {
    "ph":               (4.0,  10.0),   # Limites réalistes
    "Hardness":         (0,    500),
    "Solids":           (0,    2000),   # ❌ Changé de 60000 à 2000
    "Chloramines":      (0,    10.0),
    "Sulfate":          (0,    600),
    "Conductivity":     (0,    800),
    "Organic_carbon":   (0,    10.0),   # ❌ Changé de 30 à 10
    "Trihalomethanes":  (0,    130),
    "Turbidity":        (0,    10.0),
}

DESCRIPTIONS = {
    "ph":               "Acidité / Alkalinité (6.5 – 8.5 OMS)",
    "Hardness":         "Dureté de l'eau en mg/L (max 300 OMS)",
    "Solids":           "Solides dissous totaux en ppm (max 500 OMS)",
    "Chloramines":      "Chloramines en ppm (max 4 OMS)",
    "Sulfate":          "Sulfates en mg/L (max 250 OMS)",
    "Conductivity":     "Conductivité électrique en μS/cm (max 400 OMS)",
    "Organic_carbon":   "Carbone organique en ppm (max 2 OMS)",
    "Trihalomethanes":  "Trihalométhanes en μg/L (max 80 OMS)",
    "Turbidity":        "Turbidité en NTU (max 4 OMS)",
}

# VALEURS TYPIQUES RÉALISTES (nettoyées)
TYPICAL_VALUES = {
    "ph": 7.2,
    "Hardness": 196.4,
    "Solids": 450.0,        # ❌ Changé de 20927 à 450
    "Chloramines": 2.5,     # ❌ Changé de 7.12 à 2.5
    "Sulfate": 250.0,
    "Conductivity": 350.0,
    "Organic_carbon": 1.5,  # ❌ Changé de 14.3 à 1.5
    "Trihalomethanes": 60.0,
    "Turbidity": 3.5
}

SLIDER_RANGES = {
    "ph":               (0.0,   14.0,  0.01),
    "Hardness":         (0.0,   500.0, 0.1),
    "Solids":           (0.0,   2000.0, 1.0),   # ❌ Changé max à 2000
    "Chloramines":      (0.0,   10.0,  0.01),
    "Sulfate":          (0.0,   600.0, 0.1),
    "Conductivity":     (0.0,   800.0, 0.1),
    "Organic_carbon":   (0.0,   10.0,  0.01),   # ❌ Changé max à 10
    "Trihalomethanes":  (0.0,   130.0, 0.1),
    "Turbidity":        (0.0,   10.0,  0.01),
}

# ─────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────
def validate_inputs(input_values):
    """Valide les valeurs saisies et retourne les avertissements"""
    warnings = []
    critical_errors = []
    
    for feat, val in input_values.items():
        # Vérification des limites physiques
        min_phys, max_phys = PLAUSIBLE_RANGES[feat]
        if val < min_phys or val > max_phys:
            critical_errors.append(
                f"**{feat}** = {val:.3f} est hors des limites réalistes "
                f"({min_phys:.1f} – {max_phys:.1f})"
            )
        
        # Alerte sur dépassement normes OMS
        min_oms, max_oms = WHO_NORMS[feat]
        if val < min_oms or val > max_oms:
            warnings.append(
                f"**{feat}** = {val:.3f} dépasse la norme OMS "
                f"({min_oms:.1f} – {max_oms:.1f})"
            )
    
    return critical_errors, warnings

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
st.sidebar.warning("⚠️ Plages réalistes : pH=4-10, Solids<2000, Organic_carbon<10")

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
    # Validation des entrées
    critical_errors, warnings = validate_inputs(input_values)
    
    if critical_errors:
        st.error("### ⚠️ ERREUR CRITIQUE - Valeurs impossibles")
        for err in critical_errors:
            st.error(err)
        st.warning("Veuillez corriger les valeurs saisies (pH entre 4 et 10, Solids < 2000 mg/L, etc.)")
    else:
        # Afficher les avertissements
        if warnings:
            st.warning("### ⚠️ Paramètres hors normes OMS détectés")
            for warn in warnings:
                st.warning(warn)
        
        # PRÉDICTION CORRIGÉE : TOUJOURS SCALER !
        X_input = pd.DataFrame([input_values])
        
        # ✅ CORRECTION : Toujours scaler car le meilleur modèle est Voting Ensemble
        # (contient une régression logistique qui nécessite scaling)
        try:
            X_scaled = scaler.transform(X_input)
        except:
            # Fallback si le scaler attend les mêmes features
            X_input_reindexed = X_input.reindex(columns=FEATURES, fill_value=0)
            X_scaled = scaler.transform(X_input_reindexed)
        
        prob = model.predict_proba(X_scaled)[0][1]
        prediction = int(prob >= threshold)

        # ── Résultat principal ──
        col1, col2, col3 = st.columns([1, 1, 1])

        with col2:
            if prediction == 1:
                st.success("### ✅ EAU POTABLE")
                st.metric("Probabilité de potabilité", f"{prob:.1%}")
            else:
                st.error("### ❌ EAU NON POTABLE")
                st.metric("Probabilité de potabilité", f"{prob:.1%}")

            st.progress(prob, text=f"Score : {prob:.3f} (seuil = {threshold:.2f})")

        st.markdown("---")

        # ── Paramètres hors norme OMS ──
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
            if ok_params:
                st.markdown("**🟢 Paramètres conformes :**")
                for feat, val in ok_params[:5]:  # Limiter l'affichage
                    lo, hi = WHO_NORMS[feat]
                    st.success(f"{feat} = {val:.3f}")

        st.markdown("---")

        # ── SHAP local (version corrigée) ──
        st.subheader("🧠 Explication de la décision (SHAP)")
        
        try:
            # Préparer les données pour SHAP
            X_input_df = pd.DataFrame(X_scaled, columns=FEATURES)
            
            # Sélectionner l'explainer selon le modèle
            if hasattr(model, 'get_booster'):  # XGBoost
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_input_df)
                # Gérer le cas multi-classes
                if isinstance(shap_values, list):
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif hasattr(model, 'estimators_'):  # Random Forest ou Voting
                # Pour Voting, prendre le premier arbre
                if hasattr(model, 'named_estimators_'):
                    base_model = model.named_estimators_.get('rf') or model.named_estimators_.get('xgb')
                    if base_model and hasattr(base_model, 'get_booster'):
                        explainer = shap.TreeExplainer(base_model)
                        shap_values = explainer.shap_values(X_input_df)
                    else:
                        raise ValueError("Modèle non supporté pour SHAP")
                else:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_input_df)
            else:
                raise ValueError(f"SHAP non supporté pour {type(model)}")
            
            # Extraire les valeurs pour la classe positive
            if len(shap_values.shape) == 3:  # (n_samples, n_features, n_classes)
                shap_values_class = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:  # (n_samples, n_features)
                shap_values_class = shap_values[0]
            else:
                shap_values_class = shap_values[0]
            
            # Créer le graphique des contributions
            fig, ax = plt.subplots(figsize=(10, 5))
            sorted_idx = np.argsort(np.abs(shap_values_class))[::-1]
            top_features = [FEATURES[i] for i in sorted_idx[:8]]
            top_shap = shap_values_class[sorted_idx[:8]]
            
            colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in top_shap]
            bars = ax.barh(top_features, top_shap, color=colors)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
            ax.set_xlabel("Impact SHAP (positif → Potable, négatif → Non potable)")
            ax.set_title("Contribution des paramètres à la décision")
            
            # Ajouter les valeurs
            for bar, val in zip(bars, top_shap):
                ax.text(val + (0.005 if val >= 0 else -0.005), 
                       bar.get_y() + bar.get_height()/2,
                       f"{val:.3f}", va='center', 
                       ha='left' if val >= 0 else 'right', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.info(f"💡 Analyse SHAP non disponible : {str(e)[:200]}")
            st.markdown("""
            **Interprétation simplifiée :**
            - Plus la probabilité est proche de 100%, plus l'eau est susceptible d'être potable
            - Les paramètres marqués "🔴 Hors norme" sont les principales raisons de non-potabilité
            """)

        # ── Tableau récapitulatif ──
        st.markdown("---")
        st.subheader("📊 Tableau des valeurs saisies")
        recap = pd.DataFrame({
            "Paramètre": FEATURES,
            "Valeur": [input_values[f] for f in FEATURES],
            "Norme OMS min": [WHO_NORMS[f][0] for f in FEATURES],
            "Norme OMS max": [WHO_NORMS[f][1] for f in FEATURES],
            "Statut": ["🔴 Hors norme" if (input_values[f] < WHO_NORMS[f][0] or input_values[f] > WHO_NORMS[f][1])
                       else "🟢 Conforme" for f in FEATURES]
        })
        st.dataframe(recap, use_container_width=True, hide_index=True)

else:
    # ── Écran d'accueil ──
    st.info("👈 Saisissez les paramètres dans le panneau gauche, puis cliquez sur **Analyser l'échantillon**.")

    st.markdown("### Comment utiliser cet outil ?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Mesurer**")
        st.markdown("Prélevez un échantillon d'eau et mesurez les 9 paramètres physico-chimiques.")
    with col2:
        st.markdown("**2️⃣ Saisir**")
        st.markdown("Entrez les valeurs dans le panneau de gauche. Les valeurs pré-remplies sont des valeurs réalistes.")
    with col3:
        st.markdown("**3️⃣ Analyser**")
        st.markdown("Cliquez sur **Analyser** — l'outil indique si l'eau est potable et explique la décision.")

    st.markdown("---")
    st.markdown(f"""
    **Informations techniques**  
    - Modèle : `{model_name}`  
    - Seuil de décision : `{threshold:.2f}`  
    - ✅ **Correction 2026** : Valeurs d'entrée réalistes (Solids < 2000, pH 4-10)
    """)