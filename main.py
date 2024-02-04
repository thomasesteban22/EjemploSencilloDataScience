import pandas as pd
from sklearn.linear_model import LinearRegression

# Cargar datos
casas = pd.read_csv("casas.csv")

# Seleccionar variables
X = casas[['metros_cuadrados', 'habitaciones', 'ubicacion']]

# Transformar variable categórica
ubicacion_map = {'urbano': 1, 'suburbano': 2, 'rural': 3}
X['ubicacion'] = X['ubicacion'].map(ubicacion_map)

# Eliminar valores nulos
X = X.dropna()

# Definir variable objetivo
y = casas['precio']

# Entrenar modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Evaluar modelo
score = modelo.score(X, y)
predicciones = modelo.predict(X)

# Imprimir resultados
print("R^2:", score)
print("Predicciones:", predicciones)
