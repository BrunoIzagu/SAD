import pandas as pd
import ast

def procesar_reviews_con_fecha(csv_entrada, csv_salida):
    # Cargar CSV original
    df = pd.read_csv(csv_entrada)

    # Lista para almacenar filas con 'id', 'fecha' y 'review'
    data_procesada = []

    for idx, row in df.iterrows():
        id_listing = row['_id']
        raw_reviews = row.get('reviews', '')

        if pd.isna(raw_reviews) or raw_reviews.strip() in ('', '{}', '[]'):
            continue  # Sin reviews

        try:
            # Convertir string a lista o dict de reseñas
            parsed_reviews = ast.literal_eval(raw_reviews)

            # Si es dict único, convertirlo en lista
            if isinstance(parsed_reviews, dict):
                parsed_reviews = [parsed_reviews]

            for review in parsed_reviews:
                comment = review.get('comments', '').strip()
                fecha = review.get('date', '').strip()
                
                if comment and fecha:
                    data_procesada.append({
                        'id': id_listing,
                        'fecha': fecha,
                        'review': comment
                    })

        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            continue

    # Crear nuevo DataFrame y exportar a CSV
    df_final = pd.DataFrame(data_procesada)
    df_final.to_csv(csv_salida, index=False)
    print(f"\nArchivo procesado guardado en: {csv_salida}")
    print(f"Total de comentarios procesados: {len(df_final)}")

if __name__ == '__main__':
    procesar_reviews_con_fecha('Brazil_translated.csv', 'id_date_rev_Br.csv')

