import pandas as pd
import numpy as np
import argparse

def combinar_predicciones_regresion(csv1, csv2, mse1, mse2, salida_csv):
    # Cargar predicciones
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Verificar que tengan las columnas correctas
    if 'apartment_id' not in df1.columns or 'RatingPrediccion_scaled' not in df1.columns:
        raise ValueError("CSV1 debe tener las columnas 'apartment_id' y 'RatingPrediccion_scaled'")
    if 'apartment_id' not in df2.columns or 'RatingPrediccion_scaled' not in df2.columns:
        raise ValueError("CSV2 debe tener las columnas 'apartment_id' y 'RatingPrediccion_scaled'")

    # Verificar que los IDs coincidan
    if not df1['apartment_id'].equals(df2['apartment_id']):
        raise ValueError("Los apartment_id de los dos CSV no coinciden o están en distinto orden")

    # Calcular pesos
    epsilon = 1e-8
    w1 = 1 / (mse1 + epsilon)
    w2 = 1 / (mse2 + epsilon)
    total = w1 + w2
    w1 /= total
    w2 /= total

    # Combinar predicciones correctamente
    pred_comb = w1 * df1['RatingPrediccion_scaled'] + w2 * df2['RatingPrediccion_scaled']

    # Guardar resultado
    df_resultado = pd.DataFrame({
        'apartment_id': df1['apartment_id'],
        'RatingPrediccion_scaled': pred_comb
    })

    df_resultado.to_csv(salida_csv, index=False)
    print(f"Predicciones combinadas guardadas en {salida_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combina predicciones de regresión ponderadas según MSE.")
    parser.add_argument("--csv1", type=str, required=True, help="CSV del modelo 1 (con columnas apartment_id y RatingPrediccion_scaled)")
    parser.add_argument("--csv2", type=str, required=True, help="CSV del modelo 2 (con columnas apartment_id y RatingPrediccion_scaled)")
    parser.add_argument("--mse1", type=float, required=True, help="MSE del modelo 1")
    parser.add_argument("--mse2", type=float, required=True, help="MSE del modelo 2")
    parser.add_argument("--out", type=str, required=True, help="Archivo de salida")

    args = parser.parse_args()

    combinar_predicciones_regresion(args.csv1, args.csv2, args.mse1, args.mse2, args.out)

