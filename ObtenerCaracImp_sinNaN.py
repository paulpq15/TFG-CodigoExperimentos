# Importamos las bibliotecas
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Cargamos la base de datos
data = pd.read_csv('AcousticParametersGroupWords_sinNaN.csv')

# Seleccionar columnas por índice
X = data.iloc[:, 7:]

# Obtenemos Y
y = data['Emotion_Label']

# Aplicamos el proceso para obtener las caracteristicas mas importantes
caracteristicas_importantes = mutual_info_classif(X, y)

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