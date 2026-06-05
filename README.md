# 💧 Projet H2 — Prédiction de la potabilité de l'eau

## Structure des fichiers

```
├── water_potability.csv          ← Dataset Kaggle (à télécharger)
├── requirements.txt              ← Dépendances Python
│
├── 1_eda.py                      ← Étape 1 : Exploration & nettoyage
├── 2_modeling.py                 ← Étape 2 : Entraînement des modèles
├── 3_optimization.py             ← Étape 3 : Seuil, SHAP, validation croisée
├── app.py                        ← Étape 4 : Application Streamlit
│
├── water_potability_clean.csv    ← Généré par 1_eda.py
├── best_model.pkl                ← Généré par 2_modeling.py
├── scaler.pkl                    ← Généré par 2_modeling.py
├── best_model_name.txt           ← Généré par 2_modeling.py
└── optimal_threshold.npy         ← Généré par 3_optimization.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Ordre d'exécution

### 1. Télécharger le dataset
- URL : https://www.kaggle.com/datasets/adityakadiwal/water-potability
- Placer `water_potability.csv` dans le même dossier que les scripts

### 2. EDA & Nettoyage
```bash
python 1_eda.py
```
→ Génère `water_potability_clean.csv` + graphiques EDA

### 3. Entraînement des modèles
```bash
python 2_modeling.py
```
→ Génère `best_model.pkl`, `scaler.pkl`, `best_model_name.txt` + graphiques

### 4. Optimisation & SHAP
```bash
python 3_optimization.py
```
→ Génère `optimal_threshold.npy` + graphiques SHAP

### 5. Lancer l'application Streamlit
```bash
streamlit run app.py
```
→ Ouvre l'application sur https://projet-potabilite-eau-667gnpnfdko4imxs5vgx8a.streamlit.app/

## Ce que fait chaque étape

| Étape | Script | Output |
|-------|--------|--------|
| EDA | `1_eda.py` | Distributions, corrélations, boxplots, dataset nettoyé |
| Modélisation | `2_modeling.py` | 4 modèles comparés (LR, RF, XGBoost, SVM), courbes ROC |
| Optimisation | `3_optimization.py` | Seuil optimal, SHAP features importance, CV 5-fold |
| Application | `app.py` | Interface Streamlit terrain |

