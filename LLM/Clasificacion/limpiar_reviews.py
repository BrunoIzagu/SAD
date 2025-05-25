import pandas as pd
import re

def limpiar_review(review):
    if pd.isna(review):
        return None

    original = review.strip()

    # Detectar patrones de texto meta/sistema/traducción automática
    patrones_invalidos = [
        "translation is", 
        "direct translation", 
        "I will return the text", 
        "no discernible meaning",
        "the provided text", 
        "string of disconnected", 
        "appears to be", 
        "simply the original text", 
        "Given the nature of the input"
    ]
    lower = original.lower()
    if any(pat in lower for pat in patrones_invalidos):
        return None

    # Limpiar símbolos basura al principio y final
    review = re.sub(r"^[\s\.,/\\;:\-–—_~|+=(){}\[\]*&^%$#@!¡¿?<>\"']+", '', original)
    review = re.sub(r"[\s\.,/\\;:\-–—_~|+=(){}\[\]*&^%$#@!¡¿?<>\"']+$", '', review)

    # Verificar si queda algo útil
    contenido_util = re.sub(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ]", '', review).strip()
    if len(contenido_util.split()) < 2:
        return None

    # Rechazar si más del 70% de caracteres no son letras
    total_chars = len(review)
    alphas = len(re.findall(r'[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]', review))
    if total_chars > 0 and alphas / total_chars < 0.3:
        return None

    return review


# Leer archivo original
df = pd.read_csv("id_date_rev_Br.csv")

# Guardamos una copia original por si quieres comparar luego
df['review_original'] = df['review']

# Aplicar limpieza
df['review'] = df['review'].apply(limpiar_review)

# Mostrar qué filas fueron modificadas o eliminadas
modificadas = df[(df['review'].notna()) & (df['review_original'] != df['review'])]
eliminadas = df[df['review'].isna()]

print("🔧 Filas modificadas (limpiadas):", modificadas.index.tolist())
print("❌ Filas eliminadas (basura):", eliminadas.index.tolist())

# Eliminar las filas donde la review quedó vacía
df_final = df.dropna(subset=['review'])

# Conservar solo columnas originales: id, fecha, review
df_final = df_final[['id', 'fecha', 'review']]

# Guardar en nuevo archivo
df_final.to_csv("reviews_limpias_Br.csv", index=False)

