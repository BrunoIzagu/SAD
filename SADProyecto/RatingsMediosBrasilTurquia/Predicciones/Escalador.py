import pandas as pd
import argparse
import os
from sklearn.preprocessing import MinMaxScaler

parse = argparse.ArgumentParser(description=" Creador de csv de predicciones")
parse.add_argument("-f", "--file", help="Nombre del file entrada", required=True)
args = parse.parse_args()

# Leer el archivo CSV original
df = pd.read_csv(args.file)

# Calcular la media del RatingPrediccion por apartment_id
mean_ratings = df.groupby('apartment_id')['RatingPrediccion'].mean().reset_index()

# Escalar la columna de medias al rango [0, 1]
scaler = MinMaxScaler()
mean_ratings['RatingPrediccion_scaled'] = scaler.fit_transform(mean_ratings[['RatingPrediccion']])

# Crear nombre de salida
filename_out = os.path.splitext(args.file)[0] + '_escalados.csv'

# Guardar nuevo CSV
mean_ratings[['apartment_id', 'RatingPrediccion_scaled']].to_csv(filename_out, index=False)

print(f"CSV generado como '{filename_out}'")
