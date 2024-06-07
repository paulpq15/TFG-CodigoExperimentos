# Este archivo sirve para obtener los valores de las caracteristicas
# mas importantes de la base de datos

# Importamos las bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_classif

# Cargamos la base de datos
data = pd.read_csv('AcousticParametersGroupWords.csv')

# Seleccionar columnas por índice
X = data.iloc[:, 7:]
imputer = SimpleImputer(strategy='mean')
# Obtenemos X de la siguiente manera para quitar los errores NaN
X_imputed = imputer.fit_transform(X)

# Obtenemos Y
y = data['Emotion_Label']

# Aplicamos el proceso para obtener las caracteristicas mas importantes
caracteristicas_importantes = mutual_info_classif(X_imputed, y)

lista_Carac = []

for i in range(10):
    lista_Carac.append(caracteristicas_importantes)

carac_final = lista_Carac[0].copy()

# Sumar cada elemento de un array con el otro de otro array
for i in range(1, len(lista_Carac)):
    carac_final += lista_Carac[i]

# Realizamos la media
media_carac = carac_final / 10

# Mostramos las caracteristicas mas importantes
print("Las caracteristicas mas importantes son:", media_carac)