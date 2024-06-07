import pandas as pd

# Cargar la base de datos
data = pd.read_csv('AcousticParametersGroupWords.csv')

print(data)

# Identificar los valores NaN en el DataFrame
nan_values = data.isna()

# Imprimir los valores NaN por columna
print("Valores NaN por columna:")
print(nan_values.sum())

# Imprimir las filas que contienen al menos un NaN
print("\nFilas que contienen al menos un NaN:")
print(data[data.isna().any(axis=1)])