import pandas as pd
import json
import sys
import argparse
import textwrap
from langchain_ollama.llms import OllamaLLM
from langdetect import detect
import ftfy
import ast
import os

def load_data(file):
    """Carga un archivo CSV y maneja errores."""
    try:
        return pd.read_csv(file)
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        sys.exit(1)

def estimate_tokens(text):
    """Estima la cantidad de tokens en un texto basado en el número de palabras."""
    return int(len(text.split()) * 1.3)

def should_translate(comment):
    """Verifica si el comentario está en otro idioma distinto al inglés y debe ser traducido."""
    try:
        if not comment or len(comment.strip()) == 0:
            return False
        lang = detect(comment)
        print(f"Idioma detectado: {lang}")
        return lang != 'en'  # Traducir si NO está en inglés
    except Exception as e:
        print(f"Error en detección de idioma: {e}")
        return False

def translate_comment(comment, model, max_tokens=800):
    """Traduce un comentario al inglés, dividiendo por párrafos y fragmentos si es necesario."""
    if not comment or len(comment.strip()) == 0:
        return ""

    paragraphs = [p.strip() for p in comment.split('\n') if p.strip()]
    if not paragraphs:
        paragraphs = [comment.strip()]

    translated_paragraphs = []

    for para in paragraphs:
        if estimate_tokens(para) > max_tokens:
            chunks = textwrap.wrap(para, width=300)
        else:
            chunks = [para]

        translated_chunks = []
        for chunk in chunks:
            prompt = f"""Translate this review into natural, fluent English. Maintain the exact meaning, including any negative comments or criticisms. Pay special attention to:
1. Preserving negations (like "not", "no", "isn't")
2. Accurately translating descriptions of location, cleanliness, privacy, and noise
3. Keeping all details about the host and property
4. Correctly handling place names and cultural references without mistranslating them
5. If the input text is extremely short (such as a single word or a name), and it does not form a complete or meaningful review on its own, do not try to interpret or expand it. Instead, return the text exactly as it was provided, without adding anything.
Only output the translated text with no additional commentary. Do not add any explanations, introductions, or conclusions:

{chunk}"""
            print(">> Texto enviado al modelo:")
            print(chunk)
            print("-" * 60)
            response = model.invoke(prompt).strip()
            translated_chunks.append(response)
            print("<< Traducción generada:")
            print(response)
            print("=" * 60)

        translated_para = " ".join(translated_chunks)
        translated_paragraphs.append(translated_para)

    return '\n\n'.join(translated_paragraphs)

def parse_reviews(reviews_data):
    if isinstance(reviews_data, str):
        try:
            return json.loads(reviews_data)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(reviews_data)
            except:
                print(f"No se pudo parsear: {reviews_data[:100]}...")
                return None
    else:
        return reviews_data

def process_reviews(df, model, max_rows):
    processed_rows = 0

    for i in range(len(df)):
        if max_rows is not None and processed_rows >= max_rows:
            print(f"Se alcanzó el límite de {max_rows} filas procesadas. Deteniendo.")
            break

        reviews_value = df.at[i, 'reviews']

        print(f"\nFila {i}, tipo de reviews: {type(reviews_value)}")
        print(f"Contenido: {str(reviews_value)[:150]}...")

        parsed_reviews = parse_reviews(reviews_value)

        if parsed_reviews is None:
            continue

        row_processed = False

        if isinstance(parsed_reviews, dict):
            comment = parsed_reviews.get('comments', '')
            print(f"Comentario encontrado (dict): {comment[:50]}...")
            if comment and should_translate(comment):
                print(f"Procesando fila {i}, comentario único")
                translated = translate_comment(comment, model)
                parsed_reviews['comments'] = translated
                df.at[i, 'reviews'] = parsed_reviews
                row_processed = True

        elif isinstance(parsed_reviews, list) and len(parsed_reviews) > 0:
            any_translated = False

            for j, review in enumerate(parsed_reviews):
                if isinstance(review, dict):
                    comment = review.get('comments', '')
                    print(f"Comentario encontrado (list[{j}]): {comment[:50]}...")
                    if comment and should_translate(comment):
                        print(f"Procesando fila {i}, comentario {j}")
                        translated = translate_comment(comment, model)
                        review['comments'] = translated
                        any_translated = True

            if any_translated:
                df.at[i, 'reviews'] = parsed_reviews
                row_processed = True

        if row_processed:
            processed_rows += 1
            print(f"Fila {i} procesada. Total de filas procesadas: {processed_rows}/{max_rows if max_rows else '∞'}")

    print(f"Se procesaron {processed_rows} filas con comentarios.")
    return df

def main():
    parser = argparse.ArgumentParser(description='Traducción de comentarios de Airbnb')
    parser.add_argument('--model', type=str, default='gemma2:2b', help='Modelo de Ollama')
    parser.add_argument('--max-rows', type=int, default=None, help='Número máximo de filas a procesar (opcional)')
    parser.add_argument('--dataset', type=str, required=True, help='Nombre del archivo CSV (Brazil_processed.csv o Turkey_processed.csv)')
    args = parser.parse_args()

    dataset_name = args.dataset.strip()
    if dataset_name not in ['Brazil_processed.csv', 'Turkey_processed.csv']:
        print("Error: El nombre del dataset debe ser 'Brazil_processed.csv' o 'Turkey_processed.csv'")
        sys.exit(1)

    output_name = dataset_name.replace('_processed.csv', '_translated.csv')

    print("Cargando datos...")
    dataset = load_data(dataset_name)
    print(f"Datos cargados: {len(dataset)} filas")

    print(f"\nInicializando modelo {args.model}...")
    model = OllamaLLM(
        model=args.model,
        temperature=0.2,
        num_predict=256,
        top_k=40,
        top_p=0.9
    )

    print("\nProcesando comentarios...")
    translated_df = process_reviews(dataset, model, max_rows=args.max_rows)

    print("\nGuardando resultados...")
    translated_df.to_csv(output_name, index=False)
    print(f"Archivo guardado como '{output_name}'")

if __name__ == "__main__":
    main()

