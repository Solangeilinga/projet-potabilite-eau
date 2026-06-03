"""
PROJET H2 — Prédiction de la potabilité de l'eau
ÉTAPE 1 : Exploration & Nettoyage des données (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

# ─────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────
df = pd.read_csv("water_potability.csv")

print("=" * 60)
print("APERÇU GÉNÉRAL")
print("=" * 60)
print(f"Shape : {df.shape}")
print(f"\nTypes :\n{df.dtypes}")
print(f"\nValeurs manquantes :\n{df.isnull().sum()}")
print(f"\nPourcentage manquants :\n{df.isnull().mean().round(3) * 100}")
print(f"\nDistribution cible :\n{df['Potability'].value_counts()}")
print(f"\nStats descriptives :\n{df.describe().round(2)}")

# ─────────────────────────────────────────
# 2. VISUALISATIONS
# ─────────────────────────────────────────
features = [c for c in df.columns if c != "Potability"]

# 2a. Distributions par classe
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat in enumerate(features):
    for label, grp in df.groupby("Potability"):
        axes[i].hist(grp[feat].dropna(), bins=40, alpha=0.6,
                     label=f"{'Potable' if label == 1 else 'Non potable'}")
    axes[i].set_title(feat)
    axes[i].legend()
plt.suptitle("Distributions des features par classe", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n Sauvegardé : eda_distributions.png")

# 2b. Heatmap corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=120)
plt.close()
print(" Sauvegardé : eda_correlation.png")

# 2c. Boxplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, feat in enumerate(features):
    df.boxplot(column=feat, by="Potability", ax=axes[i])
    axes[i].set_title(feat)
    axes[i].set_xlabel("Potabilité (0=Non / 1=Oui)")
plt.suptitle("Boxplots par classe")
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=120)
plt.close()
print(" Sauvegardé : eda_boxplots.png")

# 2d. Déséquilibre de classe
plt.figure(figsize=(5, 4))
df["Potability"].value_counts().plot(kind="bar", color=["salmon", "steelblue"])
plt.xticks([0, 1], ["Non potable (0)", "Potable (1)"], rotation=0)
plt.title("Déséquilibre des classes")
plt.ylabel("Nombre d'échantillons")
plt.tight_layout()
plt.savefig("eda_class_balance.png", dpi=120)
plt.close()
print(" Sauvegardé : eda_class_balance.png")

# ─────────────────────────────────────────
# 3. IMPUTATION DES VALEURS MANQUANTES
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("IMPUTATION — KNN (k=5)")
print("=" * 60)

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

print(f"Valeurs manquantes après imputation : {df_imputed.isnull().sum().sum()}")
df_imputed["Potability"] = df_imputed["Potability"].round().astype(int)

# Sauvegarde du dataset propre
df_imputed.to_csv("water_potability_clean.csv", index=False)
print("Dataset nettoyé sauvegardé : water_potability_clean.csv")
print(f"Shape final : {df_imputed.shape}")
