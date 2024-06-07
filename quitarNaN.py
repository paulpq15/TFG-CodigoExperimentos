import pandas as pd

# Lee el archivo CSV
data = pd.read_csv('AcousticParametersGroupWords.csv')

# Elimina las filas que contienen valores NaN
data = data.dropna()

# Guarda el DataFrame modificado en un nuevo archivo CSV
data.to_csv('AcousticParametersGroupWords_sinNaN.csv', index=False)

gooddata = pd.read_csv('AcousticParametersGroupWords_sinNaN.csv')

print(gooddata)