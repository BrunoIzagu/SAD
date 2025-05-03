import pandas as pd
import argparse

parse = argparse.ArgumentParser(description=" Creador de csv de predicciones")
parse.add_argument("-f1", "--file1", help="Nombre del file entrada con los valores verdaderos  ", required=True)
parse.add_argument("-f2", "--file2", help="Nombre del file entrada con las predicciones ", required=True)
args = parse.parse_args()

# Leer el nuevo CSV
df_new = pd.read_csv(str(args.file1))  # este debe tener las columnas _id y review_scores_value

# Leer el output.csv con los valores escalados
df_scaled = pd.read_csv(str(args.file2))  # este tiene apartment_id y RatingPrediccion_scaled

# Renombrar 'apartment_id' para que coincida con '_id'
df_scaled = df_scaled.rename(columns={'apartment_id': '_id'})

# Unir los dataframes por _id
df_merged = pd.merge(df_new, df_scaled, on='_id', how='left')

# Multiplicar la columna escalada por 10
df_merged['RatingPrediccion_scaled'] = df_merged['RatingPrediccion_scaled'] * 10

# Guardar el nuevo CSV
df_merged.to_csv('final_output.csv', index=False)

print("Archivo generado como 'final_output.csv'")

