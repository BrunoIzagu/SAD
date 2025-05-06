import pandas as pd
import argparse
import os
from sklearn.preprocessing import MinMaxScaler

parse = argparse.ArgumentParser(description=" Creador de csv de predicciones")
parse.add_argument("-f", "--file", help="Nombre del file entrada", required=True)
args = parse.parse_args()

# Leer el archivo CSV original
df = pd.read_csv(args.file, decimal=".")



# Calcular la media del RatingPrediccion por apartment_id
# Convertir a número real y eliminar valores inválidos
df['RatingPrediccion'] = pd.to_numeric(df['RatingPrediccion'], errors='coerce')
df = df.dropna(subset=['RatingPrediccion'])
mean_ratings = df.groupby('apartment_id')['RatingPrediccion'].mean().reset_index()

# Escalar la columna de medias al rango [0, 1]
scaler = MinMaxScaler()
mean_ratings['RatingPrediccion_scaled'] = scaler.fit_transform(mean_ratings[['RatingPrediccion']])
print(mean_ratings.describe())
print(mean_ratings['RatingPrediccion'].describe())
# Ver los 10 apartment_id con las medias más altas
print(mean_ratings.nlargest(10, 'RatingPrediccion'))

# Ver los 10 apartment_id con las medias más bajas
print(mean_ratings.nsmallest(10, 'RatingPrediccion'))



# Crear nombre de salida
filename_out = os.path.splitext(args.file)[0] + '_escalados.csv'

# Guardar nuevo CSV
mean_ratings[['apartment_id', 'RatingPrediccion_scaled']].to_csv(
    filename_out,
    index=False,
    sep=',',           # separador de columnas por coma
    decimal='.',       # punto como separador decimal
    float_format='%.6f'
)

print(f"CSV generado como '{filename_out}'")
