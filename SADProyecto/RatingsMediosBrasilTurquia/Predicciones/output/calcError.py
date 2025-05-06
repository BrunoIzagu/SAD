import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
import os

# Configurar argumentos
parser = argparse.ArgumentParser(description="Añade el MSE como última fila en un CSV si no hay NaNs.")
parser.add_argument("-f", "--file", help="Archivo CSV de entrada", required=True)
args = parser.parse_args()

# Leer el archivo CSV
df = pd.read_csv(args.file)

# Verificar que las columnas existen
required_cols = ['review_scores_value', 'RatingPrediccion_scaled']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Falta la columna obligatoria: {col}")

# Filtrar filas válidas solo para el cálculo del MSE
df_valid = df.dropna(subset=required_cols)

# Calcular MSE si hay suficientes filas
if len(df_valid) == 0:
    print("No se puede calcular el MSE: todas las filas tienen valores faltantes.")
    mse = "MSE no calculado"
else:
    mse = mean_squared_error(df_valid['review_scores_value'], df_valid['RatingPrediccion_scaled'])

# Eliminar filas con cualquier NaN en el DataFrame original
df_clean = df.dropna()

# Crear una fila con el resultado
error_row = pd.DataFrame({
    'review_scores_value': ['MSE'],
    'RatingPrediccion_scaled': [mse]
})

# Añadir la fila del MSE al final
df_final = pd.concat([df_clean, error_row], ignore_index=True)

# Guardar el nuevo archivo
output_file = os.path.splitext(args.file)[0] + '_con_error.csv'
df_final.to_csv(output_file, index=False)

print(f"Archivo guardado como '{output_file}'.")

