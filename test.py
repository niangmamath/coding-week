import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns

# Importation de la data et gestion des Missing Values
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
data = pd.read_csv(url, header=0)
data = data.replace('?', pd.NA)
data = data.apply(pd.to_numeric, errors='coerce')
threshold = len(data) * 0.5  
data_cleaned = data.dropna(thresh=threshold, axis=1)

# Pour éviter le warning, forçons une copie
data_cleaned = data_cleaned.copy()

# Conversion de "Biopsy" en int via .loc pour éviter le warning
data_cleaned.loc[:, "Biopsy"] = data_cleaned["Biopsy"].astype(int)

# Sélection des colonnes numériques
numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns

# Liste des colonnes à traiter pour les outliers
columns_to_treat = [
    'Age',
    'First sexual intercourse',
]

# Seuil de skewness pour décider de la transformation log
skew_threshold = 1.0

# Pour chaque colonne à traiter
for col in columns_to_treat:
    skew_val = data_cleaned[col].skew()
    # Transformation logarithmique pour réduire l'asymétrie (utilise log1p pour gérer les zéros)
    data_cleaned[col + '_log'] = np.log1p(data_cleaned[col])
    # Appliquer la winsorisation sur la version transformée
    winsorized_values = np.array(winsorize(data_cleaned[col + '_log'], limits=(0.05, 0.05)))
    data_cleaned[col + '_log_winsorized'] = winsorized_values
    print(f"Colonne '{col}' (skewness = {skew_val:.2f}): transformation log et winsorisation appliquées.")

for col in columns_to_treat:
    # Choisir la version à vérifier :
    # Si la version log winsorisée existe, on l'utilise, sinon la version winsorisée
    if col + '_log_winsorized' in data_cleaned.columns:
        col_check = col + '_log_winsorized'
    else:
        col_check = col + '_winsorized'
    
    # Calculer Q1, Q3 et IQR
    Q1 = data_cleaned[col_check].quantile(0.25)
    Q3 = data_cleaned[col_check].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Détecter les outliers selon la méthode IQR
    outliers = data_cleaned[(data_cleaned[col_check] < lower_bound) | (data_cleaned[col_check] > upper_bound)]
    count_outliers = len(outliers)
    
    print(f"Colonne '{col_check}' : {count_outliers} outliers restants (méthode IQR)")

# Séparation des features (X) et de la cible (y)
X = data_cleaned.drop(columns=['Biopsy'])
y = data_cleaned['Biopsy']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation des valeurs manquantes dans X_train avec la médiane
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)

# Application de SMOTE sur X_train_imputed
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_imputed, y_train)

# Affichage de la répartition des classes après SMOTE
print("\nRépartition des classes après SMOTE:")
print(pd.Series(y_train_res).value_counts())

# Calcul de la matrice de corrélation
corr_matrix = data_cleaned.corr()

# Affichage de la matrice de corrélation sous forme de heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()