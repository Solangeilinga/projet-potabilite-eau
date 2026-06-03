"""
PROJET H2 — Prédiction de la potabilité de l'eau
ÉTAPE 2 : Entraînement & Comparaison des modèles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    ConfusionMatrixDisplay
)

# ─────────────────────────────────────────
# 1. CHARGEMENT & SPLIT
# ─────────────────────────────────────────
df = pd.read_csv("water_potability_clean.csv")

X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation (indispensable pour LR et SVM)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train : {X_train.shape} | Test : {X_test.shape}")
print(f"Taux potable train : {y_train.mean():.2%}")

# ─────────────────────────────────────────
# 2. DÉFINITION DES MODÈLES
# ─────────────────────────────────────────
# Ratio déséquilibre pour XGBoost
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        True   # nécessite scaling
    ),


    "Random Forest": (
        RandomForestClassifier(
            n_estimators=200,
            max_depth=12,           # limite la profondeur des arbres
            min_samples_leaf=5,     # chaque feuille doit avoir ≥ 5 exemples
            min_samples_split=10,   # un nœud doit avoir ≥ 10 exemples pour se diviser
            max_features="sqrt",    # chaque arbre voit une sous-partie des features
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        False
    ),

 
    "XGBoost": (
        XGBClassifier(
            n_estimators=300,           # plus d'arbres pour compenser le learning_rate faible
            max_depth=4,                # arbres peu profonds → moins de mémorisation
            learning_rate=0.05,         # apprentissage lent et stable
            min_child_weight=5,         # poids minimum d'un nœud fils
            subsample=0.8,              # chaque arbre utilise 80% des données
            colsample_bytree=0.8,       # chaque arbre voit 80% des features
            reg_alpha=0.1,              # régularisation L1
            reg_lambda=1.5,             # régularisation L2
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        ),
        False
    ),

    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=42),
        True
    ),
}

# ─────────────────────────────────────────
# 3. ENTRAÎNEMENT & ÉVALUATION
# ─────────────────────────────────────────
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "=" * 60)
print("ENTRAÎNEMENT DES MODÈLES")
print("=" * 60)

for name, (model, needs_scaling) in models.items():
    Xtr  = X_train_sc if needs_scaling else X_train
    Xte  = X_test_sc  if needs_scaling else X_test
    Xall = scaler.transform(X) if needs_scaling else X

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    # Métriques test
    auc   = roc_auc_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred)
    cv_f1 = cross_val_score(model, Xall, y, cv=cv, scoring="f1").mean()

    # DIAGNOSTIC OVERFITTING : score train vs score test
    train_prob = model.predict_proba(Xtr)[:, 1]
    train_auc  = roc_auc_score(y_train, train_prob)
    gap        = train_auc - auc
    overfit_flag = "overfitting" if gap > 0.10 else "ok"

    results[name] = {
        "model":         model,
        "needs_scaling": needs_scaling,
        "y_pred":        y_pred,
        "y_prob":        y_prob,
        "AUC-ROC":       auc,
        "F1":            f1,
        "CV-F1":         cv_f1,
        "Train-AUC":     train_auc,
        "Gap":           gap,
    }

    print(f"\n{'─'*40}")
    print(f" {name}")
    print(f"   Train AUC : {train_auc:.4f}")
    print(f"   Test AUC  : {auc:.4f}  ({overfit_flag} — gap={gap:.4f})")
    print(f"   F1        : {f1:.4f}")
    print(f"   CV-F1     : {cv_f1:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Non potable", "Potable"]))

# ─────────────────────────────────────────
# 4. COMPARAISON VISUELLE
# ─────────────────────────────────────────

# 4a. Tableau récap avec colonne Gap
metrics_df = pd.DataFrame({
    name: {
        "Train-AUC": v["Train-AUC"],
        "AUC-ROC":   v["AUC-ROC"],
        "Gap":       v["Gap"],
        "F1":        v["F1"],
        "CV-F1":     v["CV-F1"],
    }
    for name, v in results.items()
}).T.sort_values("AUC-ROC", ascending=False)

print("\n" + "=" * 60)
print("CLASSEMENT DES MODÈLES")
print("=" * 60)
print(metrics_df.round(4))
print("\n Gap Train/Test > 0.10 → overfitting à corriger")

# 4b. Courbes ROC
plt.figure(figsize=(8, 6))
for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={v['AUC-ROC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("Taux faux positifs")
plt.ylabel("Taux vrais positifs")
plt.title("Courbes ROC — Comparaison des modèles")
plt.legend()
plt.tight_layout()
plt.savefig("modeling_roc_curves.png", dpi=120)
plt.close()
print("\n Sauvegardé : modeling_roc_curves.png")

# 4c. Matrices de confusion
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
for i, (name, v) in enumerate(results.items()):
    cm = confusion_matrix(y_test, v["y_pred"])
    ConfusionMatrixDisplay(cm, display_labels=["Non potable", "Potable"]).plot(ax=axes[i])
    axes[i].set_title(name)
plt.suptitle("Matrices de confusion", fontsize=14)
plt.tight_layout()
plt.savefig("modeling_confusion_matrices.png", dpi=120)
plt.close()
print(" Sauvegardé : modeling_confusion_matrices.png")

# 4d. Barplot AUC train vs test (diagnostic overfitting)
fig, ax = plt.subplots(figsize=(10, 5))
x      = np.arange(len(results))
width  = 0.35
names  = list(results.keys())
train_aucs = [results[n]["Train-AUC"] for n in names]
test_aucs  = [results[n]["AUC-ROC"]   for n in names]

ax.bar(x - width/2, train_aucs, width, label="Train AUC", color="steelblue", alpha=0.8)
ax.bar(x + width/2, test_aucs,  width, label="Test AUC",  color="tomato",    alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.set_ylim(0, 1.1)
ax.axhline(0.9, color="gray", linestyle="--", alpha=0.4, label="Seuil alerte overfitting")
ax.set_ylabel("AUC-ROC")
ax.set_title("Diagnostic overfitting — Train AUC vs Test AUC")
ax.legend()
plt.tight_layout()
plt.savefig("modeling_overfitting_diagnostic.png", dpi=120)
plt.close()
print("Sauvegardé : modeling_overfitting_diagnostic.png")

# 4e. Barplot métriques globales
metrics_df[["AUC-ROC", "F1", "CV-F1"]].plot(kind="bar", figsize=(10, 5))
plt.title("Comparaison des métriques par modèle")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("modeling_metrics_comparison.png", dpi=120)
plt.close()
print("Sauvegardé : modeling_metrics_comparison.png")

# ─────────────────────────────────────────
# 5. SAUVEGARDE DU MEILLEUR MODÈLE
# ─────────────────────────────────────────
best_name = metrics_df["AUC-ROC"].idxmax()
best      = results[best_name]

print(f"\n🏆 Meilleur modèle : {best_name} (AUC={best['AUC-ROC']:.4f})")

with open("best_model.pkl", "wb") as f:
    pickle.dump(best["model"], f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("best_model_name.txt", "w") as f:
    f.write(best_name)

print("Sauvegardés : best_model.pkl | scaler.pkl | best_model_name.txt")