#!/usr/bin/env python3
import pandas as pd
import argparse
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from tqdm import tqdm

# ───────────────────────── 6 SHOTS ──────────────────────────
EXAMPLES = [
    # 0) Extremadamente negativa
    {"review": "The apartment was filthy, smelt of mold, and the host never replied to my messages.", "rating": "1"},
    # 1) Negativa
    {"review": "The place was dirty and communication was terrible.", "rating": "3"},
    # 2) Neutra‑negativa
    {"review": "It was okay overall, but the check‑in was slow and the street is very noisy at night.", "rating": "5"},
    # 3) Neutra‑positiva  ← la que queremos para el one‑shot “neutral‑pos”
    {"review": "Decent stay. Nothing special, but the location is convenient and the bed comfortable.", "rating": "7"},
    # 4) Positiva
    {"review": "Everything was clean and exactly as described. Great host and quiet neighborhood.", "rating": "9"},
    # 5) Muy positiva
    {"review": "Amazing apartment! Spotless, beautifully decorated, and the host went above and beyond. Would book again.", "rating": "10"},
]

# ───────────────────────── PROMPT ──────────────────────────
BASE_PROMPT = """\
You are an impartial Airbnb quality rater.
Rate ONLY the *single* review that follows on a **0‑10 integer scale**:

• 0 = unusable or dangerously bad  
• 5 = average / acceptable  
• 10 = exceptional in every aspect  

Consider cleanliness, accuracy, communication, location, and value **if the review mentions them**.
**Reply with JUST one integer (no text, no period).**

"""

def construir_prompt(review: str, shots: int = 0, shot_indexes: list[int] | None = None) -> str:
    """
    shot_indexes: lista de posiciones (0‑based) dentro de EXAMPLES que quieres incluir.
                  Si es None → se usan los primeros `shots` de EXAMPLES.
    """
    prompt = BASE_PROMPT

    # Determinar qué ejemplos añadir
    if shots and shot_indexes is None:
        shot_indexes = list(range(shots))
    elif shot_indexes is None:
        shot_indexes = []

    shot_indexes = shot_indexes[:shots]        # no exceder n.º de shots solicitados

    for idx in shot_indexes:
        ex = EXAMPLES[idx]
        prompt += f"Review: {ex['review']}\nRating: {ex['rating']}\n\n"

    prompt += f"Review: {review}\nRating:"
    return prompt

# ───────────────────────── EVALUACIÓN ──────────────────────────
def evaluar_reviews(
    csv_entrada='reviews_limpias_Tr.csv',
    csv_salida='rating.csv',
    model_name="gemma3:4b",
    shots: int = 0,
    oneshot_neutral_pos: bool = False,
):
    df = pd.read_csv(csv_entrada)
    ratings = []

    llm = OllamaLLM(model=model_name, temperature=0, num_predict=1, top_k=10, top_p=0.5)
    prompt_tmpl = PromptTemplate.from_template("{text}")
    chain = prompt_tmpl | llm

    print(f"📊 Evaluando {len(df)} reviews con {shots}-shot prompting usando {model_name}...\n")

    # Determinar shot_indexes globales según flags
    if shots == 1 and oneshot_neutral_pos:
        shot_indexes_global = [3]              # solo el ejemplo neutra‑positiva
    else:
        shot_indexes_global = None             # uso estándar

    for _, row in tqdm(df.iterrows(), total=len(df)):
        review = str(row['review'])
        texto_prompt = construir_prompt(review, shots, shot_indexes_global)

        try:
            respuesta = chain.invoke({'text': texto_prompt}).strip()
            rating = int(float(respuesta))
            if not 0 <= rating <= 10:
                raise ValueError("fuera de rango")
        except Exception:
            rating = None  # marca para revisión manual

        ratings.append(rating)

    df['rating'] = ratings
    df.to_csv(csv_salida, index=False, encoding='utf-8')
    print(f"\n✅ Archivo con ratings guardado en: {csv_salida}")

# ───────────────────────── CLI ──────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar reviews de Airbnb con prompting')
    parser.add_argument('--input',  default='reviews_limpias_Tr.csv', help='CSV de entrada')
    parser.add_argument('--output', default='rating.csv', help='CSV de salida')
    parser.add_argument('--model',  default='gemma3:4b', help='Modelo Ollama a usar')
    parser.add_argument('--shots',  type=int, default=0, help='0‑6 ejemplos')
    parser.add_argument('--oneshot-neutral-pos', action='store_true',
                        help='Si se usa 1‑shot, emplear el ejemplo neutra‑positiva (índice 3)')
    args = parser.parse_args()

    # Normalizar número de shots
    n_shots = max(0, min(args.shots, 6))

    evaluar_reviews(
        csv_entrada=args.input,
        csv_salida=args.output,
        model_name=args.model,
        shots=n_shots,
        oneshot_neutral_pos=args.oneshot_neutral_pos
    )

