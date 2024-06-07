# Importamos las bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargamos la base de datos
data = pd.read_csv('AcousticParametersGroupWords_sinNaN.csv')

# Preprocesamos los datos
# Caracteristicas "X" y etiquetas "Y"
X = data.drop("Emotion_Label", axis=1)
y = data["Emotion_Label"]

# Filtramos para solo tener en las caracteristicas a partir de la columna duracion
X = X.iloc[:, 6:]

# Estandarizar el conjunto de datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Número de intentos
numIntentos = 10
resultados_acurracy= []
matrices_confusion = []
reports = []

for intento in range(numIntentos):

    # Aqui habria que dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
    
    # Generamos y entrenamos el modelo kNN en teoria
    k = 10  # número de vecinos - probar 3-5-10 vecinos
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)

    # Probar si realiza predicciones correctamente
    y_pred = kNN_classifier.predict(X_test)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Intento {intento + 1} - Precisión del modelo: {accuracy:.2f}')
    resultados_acurracy.append(accuracy)

    # Mostrar otras métricas de evaluación
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print(class_report)
    reports.append(class_report)

    # Mostrar la matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Matriz de Confusión:')
    print(conf_matrix)
    matrices_confusion.append(conf_matrix) 

    # Visualizar la matriz de confusión con seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'], yticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'])
    plt.xlabel('Predicciones')
    plt.ylabel('Valores Reales')
    plt.title(f'Matriz de Confusión - Intento {intento + 1}')
    plt.show()
    
# Inicializa diccionarios para almacenar las métricas
macro_avg = {'precision': [], 'recall': [], 'f1-score': []}
weighted_avg = {'precision': [], 'recall': [], 'f1-score': []}
support = []

# Recorre los informes de clasificación y extrae las métricas
for report in reports:
    for key in macro_avg.keys():
        macro_avg[key].append(report['macro avg'][key])
        weighted_avg[key].append(report['weighted avg'][key])
    support.append(report['macro avg']['support'])
    
# Calcula los promedios de las métricas
macro_avg_mean = {key: np.mean(values) for key, values in macro_avg.items()}
weighted_avg_mean = {key: np.mean(values) for key, values in weighted_avg.items()}
support_mean = np.mean(support)

# Imprime los resultados de la media classification_report
print("Macro avg:")
print(macro_avg_mean)
print("\nWeighted avg:")
print(weighted_avg_mean)
print("\nSupport mean:")
print(support_mean)

# Calcular la media de la precisión y la matriz de confusión tras el numero de intentos
mean_accuracy = np.mean(resultados_acurracy)
print("Precisión media de los intentos:", mean_accuracy)
accuracy_round = round(mean_accuracy, 2)
print("Precisión media de los intentos redondeada:", accuracy_round)

matrix_confusion_total = np.mean(matrices_confusion, axis=0)
print('Matriz de Confusión de todos los intentos:')
print(matrix_confusion_total)
matrix_confusion_total_enteros = np.round(matrix_confusion_total).astype(int)
print('Matriz de Confusión de todos los intentos redondeada:')
print(matrix_confusion_total_enteros)

# Visualizar la matriz de confusión con seaborn
plt.figure(figsize=(6, 4))
sns.heatmap(matrix_confusion_total_enteros, annot=True, fmt='d', cmap='Blues', xticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'], yticklabels=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad'])
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Matriz de Confusión')
plt.show()
