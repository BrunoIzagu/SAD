import pandas as pd
import json
import re
import pandas as pd
import ast  # Para convertir las cadenas a diccionario

def split_by_country():
    print("Iniciando separación por país...")
    df = pd.read_csv("airbnb.csv", encoding="latin1")
    df['address'] = df['address'].apply(ast.literal_eval)
    df['country'] = df['address'].apply(lambda x: x.get('country') if isinstance(x, dict) else None)

    turkey_df = df[df['country'] == 'Turkey'].drop(columns=['country'])
    brazil_df = df[df['country'] == 'Brazil'].drop(columns=['country'])

    turkey_df.to_csv("Turkey.csv", index=False, encoding="utf-8")
    brazil_df.to_csv("Brazil.csv", index=False, encoding="utf-8")

    print(f"Filas para Turkey: {len(turkey_df)}")
    print(f"Filas para Brazil: {len(brazil_df)}")
    print("Archivos Turkey.csv y Brazil.csv generados correctamente.")


def convert_to_valid_json(s):
    """Convierte una cadena de texto a un objeto JSON válido."""
    if pd.isna(s):
        return None
    
    try:
        # Primero intentar analizar directamente (algunos ya podrían ser válidos)
        try:
            return json.loads(s)
        except:
            pass
        
        # Manejo de casos específicos con apóstrofes y caracteres problemáticos
        replacements = {
            "l\'Eixample": "LExample",
            "L\'Eixample": "LExample",
            "Ko\'olauloa": "Koolauloa",
            "L\'Antiga": "LAntiga",
            "l\'Antiga": "LAntiga",
            "l\'s Kitchen": "lsKitchen",
            "d\'en": "den",
            "l\'Arpa": "lArpa",
            "King\'s Park": "Kings Park",
            "L\'Ile": "L-ile",
            "L\'Î": "L",
            "d\'Hebron": "dHebron",
            "L\'Hospitalet": "LHospitalet"
        }
        
        for old, new in replacements.items():
            s = s.replace(old, new)
        
        # Eliminar caracteres y secuencias de escape problemáticos
        # Eliminar secuencias de escape inválidas como \x
        s = re.sub(r'\\x[0-9a-fA-F]{2}', '', s)
        # Reemplazar cualquier otra secuencia de escape inválida
        s = re.sub(r'\\[^"\\\/bfnrt]', '', s)
        # Eliminar caracteres no ASCII
        s = re.sub(r'[^\x00-\x7F]+', '', s)
        
        # Reemplazar comillas simples con dobles y valores booleanos
        s = s.replace("'", '"')
        s = s.replace("True", "true")
        s = s.replace("False", "false")
        s = s.replace("None", "null")
        
        # Manejo especial para nombres de hosts con "&" y caracteres especiales
        # Escapa comillas en valores de string que ya tienen comillas dobles
        def fix_quoted_values(match):
            return match.group(1) + match.group(2).replace('"', '\\"') + match.group(3)
        
        # Aplica regex para encontrar y escapar comillas dentro de valores
        s = re.sub(r'("host_name": ")([^"]*?)(")', fix_quoted_values, s)
        s = re.sub(r'("host_about": ")([^"]*?)(")', fix_quoted_values, s)
        s = re.sub(r'("host_location": ")([^"]*?)(")', fix_quoted_values, s)
        s = re.sub(r'("street": ")([^"]*?)(")', fix_quoted_values, s)
        s = re.sub(r'("suburb": ")([^"]*?)(")', fix_quoted_values, s)
        s = re.sub(r'("government_area": ")([^"]*?)(")', fix_quoted_values, s)
        
        # Manejo especial para & en nombres de host
        s = re.sub(r'"host_name": "([^"]*?)&([^"]*?)"', r'"host_name": "\1and\2"', s)
        
        # Intentar analizar el JSON
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            # Si falla, intentamos un último enfoque
            try:
                # Método más radical: extraer manualmente los pares clave-valor
                result = {}
                
                # Extraer pares clave-valor para strings
                string_pattern = r'"([^"]+)":\s*"([^"]*)"'
                for key, value in re.findall(string_pattern, s):
                    result[key] = value
                
                # Extraer pares clave-valor para números y booleanos
                non_string_pattern = r'"([^"]+)":\s*([\d\.]+|true|false|null)'
                for key, value in re.findall(non_string_pattern, s):
                    if value == "true":
                        result[key] = True
                    elif value == "false":
                        result[key] = False
                    elif value == "null":
                        result[key] = None
                    else:
                        try:
                            if "." in value:
                                result[key] = float(value)
                            else:
                                result[key] = int(value)
                        except:
                            result[key] = value
                
                # Extraer arrays
                arrays_pattern = r'"([^"]+)":\s*\[(.*?)\]'
                for key, array_content in re.findall(arrays_pattern, s, re.DOTALL):
                    # Si el contenido parece ser strings
                    if '"' in array_content:
                        items = re.findall(r'"([^"]*)"', array_content)
                        result[key] = items
                    # Si parecen ser números
                    else:
                        items = [x.strip() for x in array_content.split(',') if x.strip()]
                        parsed_items = []
                        for item in items:
                            if item == "true":
                                parsed_items.append(True)
                            elif item == "false":
                                parsed_items.append(False)
                            elif item == "null":
                                parsed_items.append(None)
                            else:
                                try:
                                    if "." in item:
                                        parsed_items.append(float(item))
                                    else:
                                        parsed_items.append(int(item))
                                except:
                                    parsed_items.append(item)
                        result[key] = parsed_items
                
                # Extraer objetos anidados
                nested_pattern = r'"([^"]+)":\s*\{(.*?)\}'
                for key, obj_content in re.findall(nested_pattern, s, re.DOTALL):
                    # Crear un "mini-json" y procesarlo recursivamente
                    mini_json = "{" + obj_content + "}"
                    try:
                        result[key] = convert_to_valid_json(mini_json)
                    except:
                        # Si falla, dejar como string
                        result[key] = obj_content
                
                return result
            except Exception as nested_e:
                print(f"Error en procesamiento avanzado: {nested_e}")
                return None
    
    except Exception as e:
        print(f"Error al procesar JSON: {e}")
        print("Texto JSON problemático:", s[:100], "..." if len(s) > 100 else "")
        return None

def process_json_columns(df, columns_to_process):
    """Procesa múltiples columnas JSON en un DataFrame."""
    
    for column in columns_to_process:
        if column not in df.columns:
            print(f"Columna '{column}' no encontrada en el DataFrame")
            continue
        
        print(f"Procesando columna: {column}")
        
        for i, value in enumerate(df[column]):
            try:
                df.at[i, column] = convert_to_valid_json(value)
                
                # Para mostrar progreso cuando se procesan muchos registros
                if i > 0 and i % 100 == 0:
                    print(f"  Procesados {i} registros")
                
            except Exception as e:
                print(f"Error en fila {i}, columna {column}: {e}")
                df.at[i, column] = None
        
        # Algunas estadísticas básicas para verificar el procesamiento
        valid_count = df[column].apply(lambda x: x is not None).sum()
        print(f"  Total procesado: {len(df)}, Válidos: {valid_count}")
    
    return df
    
def process_file(filename):
    print(f"\n📂 Procesando archivo: {filename}")
    df = pd.read_csv(filename)
    print(f"Dimensiones del archivo: {df.shape}")
    json_columns = ['images', 'host', 'address', 'availability', 'review_scores', 'reviews']
    df = process_json_columns(df, json_columns)
    processed_filename = filename.replace(".csv", "_processed.csv")
    df.to_csv(processed_filename, index=False)
    print(f"✅ Guardado como '{processed_filename}'")

def main():
    split_by_country()
    process_file("Brazil.csv")
    process_file("Turkey.csv")

if __name__ == "__main__":
    main()
