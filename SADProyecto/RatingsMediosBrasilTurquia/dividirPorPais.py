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


def main():
    split_by_country()

if __name__ == "__main__":
    main()
