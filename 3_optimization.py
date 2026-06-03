"""
PROJET H2 — Prédiction de la potabilité de l'eau
ÉTAPE 3 : Optimisation du seuil, SHAP & Validation croisée
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
df = pd.read_csv("water_potability_clean.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]
features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("best_model_name.txt") as f:
    model_name = f.read().strip()

print(f"Modèle chargé : {model_name}")

# Appliquer scaling si nécessaire
needs_scaling = model_name in ["Logistic Regression", "SVM"]
X_train_in = scaler.transform(X_train) if needs_scaling else X_train.values
X_test_in  = scaler.transform(X_test)  if needs_scaling else X_test.values

y_prob = model.predict_proba(X_test_in)[:, 1]

# ─────────────────────────────────────────
# 2. OPTIMISATION DU SEUIL (ASYMÉTRIQUE)
# ─────────────────────────────────────────
# Contexte : faux négatif (potable → en réalité non potable) = DANGEREUX
# On veut maximiser le recall de la classe "Non potable"
# = minimiser les faux négatifs sur la classe 0

print("\n" + "=" * 60)
print("OPTIMISATION DU SEUIL DE DÉCISION")
print("=" * 60)

thresholds = np.arange(0.1, 0.9, 0.01)
records = []

for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    records.append({
        "threshold": t,
        "f1":        f1_score(y_test, y_pred_t, zero_division=0),
        "precision": precision_score(y_test, y_pred_t, zero_division=0),
        "recall":    recall_score(y_test, y_pred_t, zero_division=0),
        # recall sur classe 0 = % de vrais "non potable" bien détectés
        "recall_0":  recall_score(y_test, y_pred_t, pos_label=0, zero_division=0),
    })

thresh_df = pd.DataFrame(records)

# Seuil optimal : maximise F1 tout en gardant recall_0 >= 0.70
candidates = thresh_df[thresh_df["recall_0"] >= 0.70]
if candidates.empty:
    optimal_row = thresh_df.loc[thresh_df["f1"].idxmax()]
else:
    optimal_row = candidates.loc[candidates["f1"].idxmax()]

optimal_threshold = optimal_row["threshold"]
print(f"\n→ Seuil optimal retenu : {optimal_threshold:.2f}")
print(f"   F1        : {optimal_row['f1']:.4f}")
print(f"   Precision : {optimal_row['precision']:.4f}")
print(f"   Recall    : {optimal_row['recall']:.4f}")
print(f"   Recall-0  : {optimal_row['recall_0']:.4f}")

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(thresh_df["threshold"], thresh_df["f1"], label="F1", color="steelblue")
ax1.plot(thresh_df["threshold"], thresh_df["recall_0"], label="Recall (Non potable)",
         color="tomato", linestyle="--")
ax1.axvline(optimal_threshold, color="green", linestyle=":", label=f"Seuil optimal={optimal_threshold:.2f}")
ax1.set_xlabel("Seuil")
ax1.set_ylabel("Score")
ax1.set_title("Optimisation du seuil de décision")
ax1.legend()
plt.tight_layout()
plt.savefig("optim_threshold.png", dpi=120)
plt.close()
print("Sauvegardé : optim_threshold.png")

# Matrice de confusion avec seuil optimal
y_pred_opt = (y_prob >= optimal_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)
ConfusionMatrixDisplay(cm, display_labels=["Non potable", "Potable"]).plot()
plt.title(f"Matrice de confusion — seuil {optimal_threshold:.2f}")
plt.tight_layout()
plt.savefig("optim_confusion_matrix.png", dpi=120)
plt.close()
print("Sauvegardé : optim_confusion_matrix.png")

# Sauvegarde du seuil
np.save("optimal_threshold.npy", optimal_threshold)
print(f"Seuil sauvegardé : optimal_threshold.npy")

# ─────────────────────────────────────────
# 3. SHAP — INTERPRÉTABILITÉ
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("ANALYSE SHAP")
print("=" * 60)

X_test_df = pd.DataFrame(X_test_in, columns=features)

# Choisir l'explainer selon le modèle
if model_name in ["Random Forest", "XGBoost"]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    # Compatibilité toutes versions de SHAP :
    # - ancienne API : liste [classe0, classe1] → shape (n, 9) chacun
    # - nouvelle API : array 3D → shape (n, 9, 2)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]           # ancienne API
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]     # nouvelle API → classe 1 (potable)
    else:
        shap_vals = shap_values
else:
    explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test_df, 100))
    shap_values = explainer.shap_values(X_test_df[:200])
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

# 3a. Beeswarm (importance globale)
plt.figure()
shap.summary_plot(shap_vals, X_test_df, show=False)
plt.title("SHAP — Importance des features")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()
print("Sauvegardé : shap_summary.png")

# 3b. Bar plot (importance moyenne)
plt.figure()
shap.summary_plot(shap_vals, X_test_df, plot_type="bar", show=False)
plt.title("SHAP — Importance moyenne par feature")
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=120, bbox_inches="tight")
plt.close()
print("Sauvegardé : shap_bar.png")

# Ranking des features
mean_shap = np.abs(shap_vals).mean(axis=0)
shap_ranking = pd.Series(mean_shap, index=features).sort_values(ascending=False)
print("\nRanking SHAP des features :")
print(shap_ranking.round(4))

# ─────────────────────────────────────────
# 4. VALIDATION CROISÉE FINALE
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION CROISÉE (5-fold stratifiée)")
print("=" * 60)

X_all = scaler.transform(X) if needs_scaling else X.values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    model, X_all, y, cv=cv,
    scoring=["f1", "roc_auc", "precision", "recall"],
    return_train_score=True
)

for metric in ["f1", "roc_auc", "precision", "recall"]:
    test_scores = cv_results[f"test_{metric}"]
    print(f"   {metric:12s} : {test_scores.mean():.4f} ± {test_scores.std():.4f}")

# Résumé final
print("\n" + "=" * 60)
print("RÉSUMÉ FINAL")
print("=" * 60)
print(f"Modèle          : {model_name}")
print(f"Seuil optimal   : {optimal_threshold:.2f}")
print(f"AUC-ROC test    : {roc_auc_score(y_test, y_prob):.4f}")
print(f"F1 (seuil opt.) : {f1_score(y_test, y_pred_opt):.4f}")
print(f"Top 3 features  : {', '.join(shap_ranking.head(3).index.tolist())}")