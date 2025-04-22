from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.evaluation import ExactMatchStringEvaluator
from datasets import load_dataset
from langdetect import detect
from concurrent.futures import ThreadPoolExecutor, as_completed
import ftfy
import pandas as pd
import sys
import argparse
import textwrap

def load_data(file):
    try:
        return pd.read_csv(file, encoding="latin1")
    except Exception as e:
        print(e)
        sys.exit(1)

def estimate_tokens(text):
    return int(len(text.split()) * 1.3)

def should_translate(comment):
    try:
        lang = detect(comment)
        return lang in ['pt', 'tr']  # portugués o turco
    except:
        return False

def translate_comment(comment, model, max_tokens=800):
    paragraphs = [p.strip() for p in comment.split('\n') if p.strip()]
    translated_paragraphs = []

    for para in paragraphs:
        if estimate_tokens(para) > max_tokens:
            chunks = textwrap.wrap(para, width=300)
        else:
            chunks = [para]

        translated_chunks = []
        for chunk in chunks:
            prompt = f"Please translate the following text from Portuguese or Turkish to English, if it is in another language skip it:\n\n{chunk}"
            print(prompt)
            response = model.invoke(prompt).strip()
            print("Num tokens del prompt:", estimate_tokens(prompt))
            print(response)
            translated_chunks.append(response)

        translated_paragraph = "\n".join(translated_chunks)
        translated_paragraphs.append(translated_paragraph)

    return '\n\n'.join(translated_paragraphs)

def process_row(n_instance):
    n, instance = n_instance
    try:
        reviews_list = eval(instance["reviews"])
    except Exception as e:
        print(f"❌ Error parsing reviews at row {n}: {e}")
        return None

    translated_reviews = []
    for review in reviews_list:
        comment = review.get("comments", "").strip()
        if comment:
            cleaned = ftfy.fix_text(comment)
            if should_translate(cleaned):
                translated = translate_comment(cleaned, model)
                review["comments"] = translated
            else:
                print("🔍 Comentario no es portugués ni turco, se omite traducción.")
        translated_reviews.append(review)

    new_row = instance.copy()
    new_row["reviews"] = str(translated_reviews)
    return new_row

# Argumentos
parser = argparse.ArgumentParser(description='casiMedicos ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='validation', help='split')
parser.add_argument('--sample', type=int, default=-1, help='sample')
parser.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
args = parser.parse_args()

# Cargar dataset y modelo
dataset = load_data(args.file)
model = OllamaLLM(model=args.model, temperature=0.7, num_predict=128, top_k=50, top_p=0.95)

# Paralelizar traducciones
translated_rows = []
limit = args.sample if args.sample != -1 else len(dataset)

with ThreadPoolExecutor(max_workers=6) as executor:
    futures = [executor.submit(process_row, (n, row)) for n, row in dataset.head(limit).iterrows()]
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            translated_rows.append(result)

# Guardar
translated_df = pd.DataFrame(translated_rows)
translated_df.to_csv("airbnb_translated.csv", index=False, encoding="utf-8")
print("✅ Traducciones guardadas en 'airbnb_translated.csv'")


