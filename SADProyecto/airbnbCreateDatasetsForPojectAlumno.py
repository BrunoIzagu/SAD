import ast

import pandas as pd
import json
import re
from collections import Counter


def convert_to_valid_json(s):
    try:
        # Step 1: Replace single quotes with double quotes
        s = s.replace("l\'Eixample","LExample")
        s = s.replace("L\'Eixample", "LExample")
        s = s.replace("Ko\'olauloa", "Koolauloa")
        s = s.replace("L\'Antiga", "LAntiga")
        s = s.replace("l\'Antiga", "LAntiga")
        s = s.replace("l\'s Kitchen","lsKitchen")
        s = s.replace("d\'en","den")
        s = s.replace("l\'Arpa","lArpa")
        s = s.replace("King\'s Park","Kings Park")
        s = s.replace("L\'Ile","L-ile")
        s = s.replace("L\'Î","L")
        s = s.replace("d\'Hebron","dHebron")
        s = s.replace("L\'Hospitalet", "LHospitalet")
        s = s.replace("'", '"')
        s = s.replace("True", "true")
        s = s.replace("False", "false")
        # Step 2: Attempt to parse JSON
        return json.loads(s)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON at position {e.pos}: {e.msg}")
        print("Problematic JSON string:", s)
        return None

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv("airbnb.csv")

# Mostrar las cabeceras del DataFrame
print(df.columns)
print(df.size)
countries=list()
for i in range(len(df)):
    el=df.loc[i].address
    dfEl = pd.DataFrame({
        'json_string': [el   ]
    })
    try:
        dfEl['json_dict'] = dfEl['json_string'].apply(convert_to_valid_json)
        countries.append(dfEl['json_dict'][0]["country"])
    except:
        print("format error")

count = Counter(countries)
print(count)


