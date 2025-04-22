from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.evaluation import ExactMatchStringEvaluator
from datasets import load_dataset
import ftfy
import pandas as pd
import sys
import argparse
import textwrap

#run "ollama pull gemma2:2b" in your terminal before running this script
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding="latin1")
        #data = pd.read_csv(file, encoding='utf-8')
        return data
    except Exception as e:
        print(e)
        sys.exit(1)
def estimate_tokens(text):
    """Estimación básica de tokens (puede variar según el modelo)."""
    return int(len(text.split()) * 1.3)

def translate_comment(comment, model, max_tokens=800):
    """
    Traduce un comentario dividiéndolo si es demasiado largo para el modelo.
    """
    paragraphs = [p.strip() for p in comment.split('\n') if p.strip()]
    translated_paragraphs = []

    for para in paragraphs:
        if estimate_tokens(para) > max_tokens:
            # Dividir en fragmentos más pequeños si es muy largo
            chunks = textwrap.wrap(para, width=300)
        else:
            chunks = [para]

        translated_chunks = []
        for chunk in chunks:
            prompt = f"Please translate the following text from Porutguese or Turkish to English, if it is in another language print it as it is:\n\n{chunk}"
            print(prompt)
            response = model.invoke(prompt).strip()
            print("Num tokens del prompt: " + str(estimate_tokens(prompt)))
            print(response)
            translated_chunks.append(response)

        translated_paragraph = "\n".join(translated_chunks)
        translated_paragraphs.append(translated_paragraph)

    return '\n\n'.join(translated_paragraphs)




parser=argparse.ArgumentParser(description='casiMedicos ollama LLM evaluation')
parser.add_argument('--model', type=str, default='gemma2:2b', help='ollama model name')
parser.add_argument('--lang', type=str, default='en', help='language')
parser.add_argument('--split', type=str, default='validation', help='split')
parser.add_argument('--sample', type=int, default=-1, help='sample')
parser.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
args=parser.parse_args()


dataset = load_data(args.file)

template = "Please translate the following text to English:\n{Sentence}"
prompt = PromptTemplate.from_template(template)
model = OllamaLLM(model=args.model, temperature=0.7, num_predict=128, top_k=50, top_p=0.95)

chain = prompt | model

evaluator = ExactMatchStringEvaluator()
ok = 0
wrongOut = 0

#dataset="/home/bruno/Escritorio/SAD/SADProyecto/airbnb.csv"
#casimed = load_dataset(dataset, args.lang) #check huggingface datasets for details
casimed=dataset
for n, instance in casimed.iterrows():
    if args.sample != -1 and n >= args.sample:
        break

    try:
        reviews_list = eval(instance["reviews"])  # Convierte string a lista de dicts, si viene como string
    except Exception as e:
        print(f"Error parsing reviews at row {n}: {e}")
        continue

    for review in reviews_list:
        comment = review.get("comments", "").strip()
        if comment:
            #print(comment)
            cleaned_comment = ftfy.fix_text(comment)
            print(cleaned_comment)
            #paragraphs = comment.split('\n')

            #translated = chain.invoke({'Sentence': paragraphs}).strip()
            translated=translate_comment(cleaned_comment,model)
            print("------------------------Traduccion: " + translated)
