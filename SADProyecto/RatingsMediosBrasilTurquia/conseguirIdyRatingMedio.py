import pandas as pd
import ast

# Cargar el CSV procesado por país (por ejemplo, Brazil)
df = pd.read_csv("Turkey.csv", encoding="utf-8")

# Convertir la columna 'review_scores' de texto a diccionario
df['review_scores'] = df['review_scores'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# Extraer el campo 'review_scores_value'
df['review_scores_value'] = df['review_scores'].apply(lambda x: x.get('review_scores_value', None))

# Crear un nuevo DataFrame con solo '_id' y 'review_scores_value'
result_df = df[['_id', 'review_scores_value']]

# Borrar valores null
result_df = result_df.dropna(subset=['review_scores_value'])

# Guardar en un nuevo CSV
result_df.to_csv("ratingsMedios_turkey.csv", index=False, encoding="utf-8")

print("✅ Archivo 'ratingsMedios_turkey.csv' generado y ordenado correctamente.")
