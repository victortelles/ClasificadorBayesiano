import pandas as pd
import numpy as np
from src.models.clasificador_bayesiano import ClasificadorBayesiano


# 0. Cargar datos
df = pd.read_csv('data/datos_evaluacion.csv')

# 1. Separacion de datos
# Mezclar dataframe
df_mezclado = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcular tamaño del conjunto de prueba (20%)
tamano_prueba = int(len(df) * 0.2)

# separacion
df_prueba = df_mezclado.iloc[:tamano_prueba]
df_entrenamiento = df_mezclado.iloc[tamano_prueba:]

# 2. Entrenamiento del clasificador bayesiano
clasificador = ClasificadorBayesiano()
clasificador.calcular_probabilidades_priori(df_entrenamiento)
clasificador.calcular_probabilidades_conjuntas(df_entrenamiento)

# 3. prediccion
predicciones = []
for _, fila in df_prueba.iterrows():
    edad = fila['Edad_bin']
    ingreso = fila['Ingreso_bin']
    clase_predicha = clasificador.predecir_clase(edad, ingreso)
    predicciones.append(clase_predicha)



# 4. Matriz de confusión y precisión manual
# 01: Inicializar la matriz de confusión
# La matriz es de 2x2
matriz = [[0, 0], [0, 0]]
# matriz[0][0] = Camina real, Camina predicho
# matriz[0][1] = Camina real, Conduce predicho
# matriz[1][0] = Conduce real, Camina predicho
# matriz[1][1] = Conduce real, Conduce predicho

# 02: Llenar la matriz de confusión
for k in range(len(df_prueba)):
    # Obtener clase real y predicha
    real = df_prueba.iloc[k]['Clase']
    pred = predicciones[k]

    # Si la predicción es una tupla, tomar solo la clase
    if isinstance(pred, tuple):
        pred = pred[0]
    # Determinar la fila (real) y columna (predicho)

    if real == 'Camina':
        fila = 0
    else:
        fila = 1

    if pred == 'Camina':
        columna = 0
    else:
        columna = 1

    # Sumar 1 en la celda correspondiente
    matriz[fila][columna] = matriz[fila][columna] + 1

# 03: Mostrar la matriz de confusion
print("Matriz de confusion:")
print("\tPredicho: Camina\tPredicho: Conduce")
print("Real: Camina\t" + str(matriz[0][0]) + "\t\t" + str(matriz[0][1]))
print("Real: Conduce\t" + str(matriz[1][0]) + "\t\t" + str(matriz[1][1]))

# 04: Calcular la precisión
# Precisión = (aciertos) / (total de ejemplos)
aciertos = matriz[0][0] + matriz[1][1]
total = matriz[0][0] + matriz[0][1] + matriz[1][0] + matriz[1][1]
if total > 0:
    precision = aciertos / total
else:
    precision = 0
print("Precisión: {:.2f}".format(precision))


