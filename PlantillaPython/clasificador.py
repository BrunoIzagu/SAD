# -*- coding: utf-8 -*-
"""
Autor: Xabier Gabiña y Ibai Sologestoa.
Script para la implementación del algoritmo de clasificación
"""

import random
import sys
import signal
import argparse
import pandas as pd
import numpy as np
import string
import pickle
import time
import json
import csv
import os
import unicodedata
import re
from colorama import Fore
# Sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report,recall_score, roc_auc_score,accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from tqdm import tqdm

# Funciones auxiliares

def signal_handler(sig, frame):
    """
    Función para manejar la señal SIGINT (Ctrl+C)
    :param sig: Señal
    :param frame: Frame
    """
    print("\nSaliendo del programa...")
    sys.exit(0)

def parse_args():
    """
    Función para parsear los argumentos de entrada
    """
    parse = argparse.ArgumentParser(description="Practica de algoritmos de clasificación de datos.")
    parse.add_argument("-m", "--mode", help="Modo de ejecución (train o test)", required=True)
    parse.add_argument("-f", "--file", help="Fichero csv (/Path_to_file)", required=True)
    parse.add_argument("-a", "--algorithm", help="Algoritmo a ejecutar (kNN, decision_tree o random_forest)", required=True)
    parse.add_argument("-p", "--prediction", help="Columna a predecir (Nombre de la columna)", required=True)
    parse.add_argument("-e", "--estimator", help="Estimador a utilizar para elegir el mejor modelo https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter", required=False, default=None)
    parse.add_argument("-c", "--cpu", help="Número de CPUs a utilizar [-1 para usar todos]", required=False, default=-1, type=int)
    parse.add_argument("-v", "--verbose", help="Muestra las metricas por la terminal", required=False, default=False, action="store_true")
    parse.add_argument("--debug", help="Modo debug [Muestra informacion extra del preprocesado y almacena el resultado del mismo en un .csv]", required=False, default=False, action="store_true")
    # Parseamos los argumentos
    args = parse.parse_args()
    
    # Leemos los parametros del JSON
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    # Juntamos todo en una variable
    for key, value in config.items():
        setattr(args, key, value)
    
    # Parseamos los argumentos
    return args
    
def load_data(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding="latin1")
        #data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)
'''def load_data1(file):
    """
    Función para cargar los datos de un fichero csv
    :param file: Fichero csv
    :return: Datos del fichero
    """
    try:
        data = pd.read_csv(file, encoding="latin1",on_bad_lines='skip')
        #data = pd.read_csv(file, encoding='utf-8')
        print(Fore.GREEN+"Datos cargados con éxito"+Fore.RESET)
        return data
    except Exception as e:
        print(Fore.RED+"Error al cargar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)'''

# Funciones para calcular métricas
def calcular_metricas_y_guardar(col_verdadero, col_predicho, dataf, archivo_salida="modeloSalida"):
    #Crea un archivo csvdonde estan los valores reales predecidos por el modelo y al final escribe las metricasque ha sacado el modelo
    # Crear carpeta output si no existe
    output = "output"
    os.makedirs(output, exist_ok=True)
    archivo_salida = os.path.join(output, archivo_salida)
    
    # Crear DataFrame con los datos proporcionados
    '''df = pd.DataFrame({
        'Verdadero': col_verdadero,
        'Predicho': col_predicho
    })'''
    df=dataf
    
    # Calcular métricas
    accuracy = accuracy_score(col_verdadero, col_predicho)
    f1 = f1_score(col_verdadero, col_predicho, average='weighted')
    recall = recall_score(col_verdadero, col_predicho, average='weighted')
    precision= precision_score(col_verdadero, col_predicho, average='weighted')
    
    # Asegurarse de que roc_auc_score sea adecuado según el tipo de problema
    if len(set(col_verdadero)) == 2:  # Solo aplica para clasificación binaria
        roc_auc = roc_auc_score(col_verdadero, col_predicho)
    else:
        roc_auc = None  # Si es multiclase, roc_auc puede no ser aplicable o requerir otro enfoque
    
    # Crear DataFrame con resultados
    resultados = pd.DataFrame({
        'Métrica': ['Accuracy', 'F1-Score', 'Recall','Precision', 'ROC-AUC'],
        'Valor': [accuracy, f1, recall, precision, roc_auc]
    })
    
    # Guardar datos y métricas en CSV
    df.to_csv(archivo_salida, header=True, index=False)  # Guardar datos de verdad y predicciones
    with open(archivo_salida, 'a') as f:  # Agregar métricas al archivo
        resultados.to_csv(f,header=True, index=False)
    
    print(f"Métricas y datos guardados en {archivo_salida}")

def calculate_fscore(y_test, y_pred):
    """
    Función para calcular el F-score
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: F-score (micro), F-score (macro)
    """
    fscore_micro = f1_score(y_test, y_pred, average='micro')
    fscore_macro = f1_score(y_test, y_pred, average='macro')
    return fscore_micro, fscore_macro

def calculate_classification_report(y_test, y_pred):
    """
    Función para calcular el informe de clasificación
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Informe de clasificación
    """
    report = classification_report(y_test, y_pred, zero_division=0)
    return report

def calculate_confusion_matrix(y_test, y_pred):
    """
    Función para calcular la matriz de confusión
    :param y_test: Valores reales
    :param y_pred: Valores predichos
    :return: Matriz de confusión
    """
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Funciones para preprocesar los datos

def select_features():
    """
    Separa las características del conjunto de datos en características numéricas, de texto y categóricas.

    Returns:
        numerical_feature (DataFrame): DataFrame que contiene las características numéricas.
        text_feature (DataFrame): DataFrame que contiene las características de texto.
        categorical_feature (DataFrame): DataFrame que contiene las características categóricas.
    """
    try:
        # Numerical features
        numerical_feature = data.select_dtypes(include=['int64', 'float64']) # Columnas numéricas
        if args.prediction in numerical_feature.columns:
            numerical_feature = numerical_feature.drop(columns=[args.prediction])
        # Categorical features
        categorical_feature = data.select_dtypes(include='object')
        categorical_feature = categorical_feature.loc[:, categorical_feature.nunique() <= args.preproceso["unique_category_threshold"]] ###################
        
        # Text features
        text_feature = data.select_dtypes(include='object').drop(columns=categorical_feature.columns)

        print(Fore.GREEN+"Datos separados con éxito"+Fore.RESET)
        
        if args.debug:
            print(Fore.MAGENTA+"> Columnas numéricas:\n"+Fore.RESET, numerical_feature.columns)
            print(Fore.MAGENTA+"> Columnas de texto:\n"+Fore.RESET, text_feature.columns)
            print(Fore.MAGENTA+"> Columnas categóricas:\n"+Fore.RESET, categorical_feature.columns)
        return numerical_feature, text_feature, categorical_feature
    except Exception as e:
        print(Fore.RED+"Error al separar los datos"+Fore.RESET)
        print(e)
        sys.exit(1)
def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def preproceso():
    global data
    if args.preproceso['Preprocesar?']=='si' :
        # Identificar las columnas 'Unnamed' en el DataFrame
        unnamed_cols = [col for col in data.columns if 'Unnamed' in col]
        # Renombrar las columnas 'Unnamed' de forma que tengan nombres más significativos
        new_names = {col: f"New_Column_{i}" for i, col in enumerate(unnamed_cols)}
        # Renombrar las columnas
        data.rename(columns=new_names, inplace=True)
         # Separamos los datos por tipos
        numerical_feature, text_feature, categorical_feature = select_features()

        # Concatenar con un delimitador
        if args.preproceso['cols_concatenar'][0]!="no":
            data=concatenar_columnas(data,args.preproceso['cols_concatenar'])

        if args.preproceso['missing_values'] == 'imputar':
            print("Imputación en proceso.")
            # Configurar la estrategia de imputación
            if args.preproceso['impute_strategy'] == 'mean':
                    imputer = SimpleImputer(strategy="mean")
                    print("Metodo es mean")
            elif args.preproceso['impute_strategy'] == 'mode':
                    imputer = SimpleImputer(strategy="most_frequent")
                    print("Metodo es moda")
            elif args.preproceso['impute_strategy'] == 'median':
                    imputer = SimpleImputer(strategy="median")
                    print("Metodo es mediana")
        

            # Obtener las columnas a imputar desde el JSON
            cols_imputar = numerical_feature
    
            # Verificar si las columnas a imputar existen en df
            cols_imputar_validas = [col for col in cols_imputar if col in data.columns]
            print(cols_imputar_validas)
            print("Valores NaN en las columnas a imputar:", data[cols_imputar_validas].isnull().sum())
            if cols_imputar_validas:  # Solo imputar si hay columnas válidas
                data[cols_imputar_validas] = imputer.fit_transform(data[cols_imputar_validas])
                print("Imputación completada.")
            else:
                print(f"⚠️ Advertencia: No se encontraron las columnas {cols_imputar} en X_train. Verifica los nombres.")
        elif args.preproceso['missing_values'] == 'drop':
            #cols_imputar = numerical_feature
            # Para cubrir varios casos raros:
            
            cols_imputar = args.preproceso['cols_imputar']


            print(f"🗑 Eliminando filas con valores faltantes en las columnas: {cols_imputar}")
            print(cols_imputar)
            print(type(cols_imputar))

            data.dropna(subset=cols_imputar, inplace=True)    
        print("Valores faltantes en data:", data.isnull().sum())

        if args.preproceso['cols_outliers'][0]!="no":
            for colum in args.preproceso['cols_outliers']:
                remove_outliers_iqr(colum)
        
        # Pasar de categorial a númerico
        if args.preproceso['cat_num?']=='si' :
            if args.preproceso['categorial_features'][0]!='no' :
                cols_cat_num=args.preproceso['categorial_features']
            else:
                cols_cat_num= categorical_feature
            cat_numerico(data,cols_cat_num)
            for col in categorical_feature:
                print(f"Valores únicos en {col}: {data[col].unique()}")

        if args.preproceso['normalize?']=='si' :
            cols_simplificar=text_feature
            for col in cols_simplificar:
                data[col] = data[col].fillna('').apply(lambda x: limpiar_texto(x))
        print(data.head())  # Esto imprimirá las primeras 5 filas de todas las columnas por defecto
        process_text(text_feature)
        
        if args.preproceso['scaling']=='standar' :
            # Escalamos los datos de forma standard
            data=escaladoEstandar(data)
        elif args.preproceso['scaling']=='absmaxmin' :
            # Escalamos los datos de forma absminmax
            data=maximum_absolute_scaling(data)
        elif args.preproceso['scaling']=='minmax' :
            # Escalamos los datos de forma minmax
            cols=data.columns
            for col in cols :
                min_max_scaling(col)
        elif args.preproceso['scaling']=='zscore' :
            # Escalamos los datos de forma zscore
            cols=data.columns
            for col in cols :
                z_score(col)
        #eliminamos cols
        if args.preproceso['cols_eliminar'][0]!= "no":
            cols_eliminar=args.preproceso['cols_eliminar']
            data = data.drop(cols_eliminar, axis=1)
        
        # Realizamos Oversampling o Undersampling
        over_under_sampling()
        print("Despues del preprocesado")  
        print(data.head())  # Esto imprimirá las primeras 5 filas de todas las columnas por defecto
        

    return data

    
def concatenar_columnas(data, columnas):
    """
    Concatenar varias columnas en un nuevo columna llamada 'concatenada'.
    
    Parámetros:
    - X_train: DataFrame de pandas que contiene las columnas a concatenar.
    - columnas: Lista de nombres de las columnas que deseas concatenar.
    - sep: Separador que se usará para concatenar las columnas (por defecto es ',').
    
    Retorna:
    - DataFrame con la columna 'concatenada' añadida.
    """
    sep=' '
    # Verificar que la lista de columnas tiene al menos dos columnas
    if len(columnas) < 2:
        raise ValueError("La lista de columnas debe contener al menos dos columnas.")
    
    # Asegurarse de que no haya valores NaN antes de concatenar
    for col in columnas:
        data[col] = data[col].fillna('')  # Reemplazar NaN por una cadena vacía
    
    # Usar str.cat() para concatenar las columnas seleccionadas
    data[columnas[0]] = data[columnas[0]].str.cat(data[columnas[1]], sep=sep)
    
    # Si hay más de dos columnas, concatenarlas
    for col in columnas[2:]:
        data[columnas[0]] = data[columnas[0]].str.cat(data[col], sep=sep)
    
    return data

def guardarModelo(clf):
    nombreModel = "nombreParAlmacenar.sav"
    with open(nombreModel, 'wb') as archivo:
        pickle.dump(clf, archivo)
    print("Modelo guardado correctamente empleando Pickle")

def cargarModelo(nombreModel):
    with open(nombreModel, 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo
def remove_outliers_iqr(column):
    """
    Elimina los outliers utilizando el método del Rango Intercuartílico (IQR).
    
    Parámetros:
    - df: DataFrame, el conjunto de datos.
    - column: str, el nombre de la columna sobre la que se aplicará la detección de outliers.
    
    Retorna:
    - df: DataFrame, el conjunto de datos sin los outliers.
    """
    global data
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definimos el límite inferior y superior para los outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtramos los valores fuera de este rango
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
#Preprocesado      
def limpiar_texto(texto):
    """
    Limpia y normaliza texto basado en configuraciones de `miJson`:
    - Convierte a minúsculas
    - Elimina acentos y caracteres especiales
    - Elimina stopwords
    - Aplica lematización
    """

    # Obtener la configuración de preprocesamiento
    opciones = miJson.get('preproceso', {}).get('normalize_vector', [])

    lemmatizer = WordNetLemmatizer()
    stop_words_list = set(stopwords.words("english")) if "stopwords" in opciones else set()

    # Si el texto es NaN, None o vacío, devolverlo sin cambios
    if texto is None or texto == "" or pd.isna(texto):
        return texto

    # 1. Convertir a minúsculas
    if "minusculas" in opciones:
        texto = texto.lower().strip()

    # 2. Quitar acentos
    if "acentos" in opciones:
        texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8')

    # 3. Eliminar caracteres especiales y números
    if "caracEsp" in opciones:
        texto = re.sub(r'[^a-z\s]', '', texto)

    # 4. Tokenizar el texto (convertir en lista de palabras)
    palabras = word_tokenize(texto, language="english") if "tokenizar" in opciones else texto.split()

    # 5. Eliminar stopwords y aplicar lematización si está activado
    if "lematizar" in opciones:
        palabras = [lemmatizer.lemmatize(p) for p in palabras if p not in stop_words_list]

    # 6. Unir palabras procesadas en una sola cadena
    texto_limpio = " ".join(palabras)

    return texto_limpio

def cat_numerico(df, columnas):
    """
    Convierte columnas categóricas a numéricas usando pd.factorize().
    
    Parámetros:
    - df: DataFrame de Pandas
    - columnas: Lista de nombres de columnas a convertir (opcional).
                Si es None, convierte todas las categóricas.

    Retorna:
    - DataFrame con las columnas convertidas a valores numéricos.
    """
    if columnas is None:
        # Si no se especifican columnas, convertir todas las categóricas
        columnas = df.select_dtypes(exclude=['number']).columns.tolist()
    else:
        # Filtrar solo las que existen en el DataFrame
        columnas = [col for col in columnas if col in df.columns]

    # Aplicar pd.factorize() a las columnas seleccionadas
    df[columnas] = df[columnas].apply(lambda x: pd.factorize(x)[0])
    
    return df

###############ESCALADO############
def z_score(v):
    """Escala una columna usando el método Z-Score, asegurando que sea numérica."""
    
    # Imprime el tipo para depuración
    print(f"Tipo de v: {type(v)}")

    # Si v es un solo valor string (no una columna), lo devolvemos sin cambios
    if isinstance(v, str):
        return v
    
    # Convertimos a Serie si es necesario
    if isinstance(v, (list, np.ndarray)):
        v = pd.Series(v)

    # Verificamos si es numérico
    if not np.issubdtype(v.dtype, np.number):  
        return v  # Retorna sin cambios si no es numérico

    return (v - v.mean()) / v.std()
def maximum_absolute_scaling(df):
    """Escala columnas numéricas con media > 60 usando el escalado máximo absoluto."""
    df_scaled = df.copy()
    for column in df_scaled.select_dtypes(include=[np.number]).columns:  # Solo numéricas
        if df_scaled[column].mean() > 60:
            df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled
def min_max_scaling(v):
    """Escala una columna usando el método Min-Max."""
    # Convertir a NumPy array asegurando que sean números
    try:
        v = np.array(v, dtype=float)  # Convierte a números si es posible
    except ValueError:
        print("Error: La columna contiene valores no numéricos.", v)
        return None  # Evita procesar datos incorrectos
    print(type(v))
    if not np.issubdtype(v.dtype, np.number):
        return v  # Retorna sin cambios si es texto
    return (v - v.min()) / (v.max() - v.min())
    return v_norm 
def escaladoEstandar(df):
    """Escala todo el DataFrame usando StandardScaler de sklearn sin modificar el original."""
    df_scaled = df.copy()  # Copia el DataFrame para evitar modificaciones en el original
    num_cols = df_scaled.select_dtypes(include=[np.number]).columns  # Selecciona solo columnas numéricas
    sc = StandardScaler()
    
    # Guardamos los tipos originales
    dtypes_originales = df_scaled[num_cols].dtypes  
    
    # Escalamos las columnas numéricas
    df_scaled[num_cols] = sc.fit_transform(df_scaled[num_cols])
    
    # Restauramos el tipo de dato original para evitar problemas
    for col in num_cols:
        if np.issubdtype(dtypes_originales[col], np.integer):
            df_scaled[col] = df_scaled[col].astype(int)  # Convierte de vuelta a enteros
    
    return df_scaled
def process_text(text_feature):
    """
    Procesa las características de texto utilizando técnicas de vectorización como TF-IDF o BOW.

    Parámetros:
    text_feature (pandas.DataFrame): Un DataFrame que contiene las características de texto a procesar.

    """
    global data
    try:
        if text_feature.columns.size > 0:
            if args.preproceso["max_arg_textProcessor"]!="no":
                max_featuresPre=args.preproceso["max_arg_textProcessor"]
            else:
                max_featuresPre=None
            if args.preproceso["text_process"] == "tf-idf":     
                tfidf_vectorizer = TfidfVectorizer(max_features=max_featuresPre)
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)

                # ✅ Índice alineado para evitar NaNs ocultos
                text_features_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=tfidf_vectorizer.get_feature_names_out(),
                    index=data.index
                )

                data = pd.concat([data, text_features_df], axis=1)
                data.drop(text_feature.columns, axis=1, inplace=True)

                print(Fore.GREEN + "Texto tratado con éxito usando TF-IDF" + Fore.RESET)

            elif args.preproceso["text_process"] == "bow":
                bow_vecotirizer = CountVectorizer()
                text_data = data[text_feature.columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
                bow_matrix = bow_vecotirizer.fit_transform(text_data)

                # También puede alinearse por si acaso
                text_features_df = pd.DataFrame(
                    bow_matrix.toarray(),
                    columns=bow_vecotirizer.get_feature_names_out(),
                    index=data.index
                )

                data = pd.concat([data, text_features_df], axis=1)
                print(Fore.GREEN + "Texto tratado con éxito usando BOW" + Fore.RESET)

            else:
                print(Fore.YELLOW + "No se están tratando los textos" + Fore.RESET)
        else:
            print(Fore.YELLOW + "No se han encontrado columnas de texto a procesar" + Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al tratar el texto"+Fore.RESET)
        print(e)
        sys.exit(1)

def over_under_sampling():
    """
    Realiza oversampling o undersampling en los datos según la estrategia especificada en args.preproceso["sampling"].
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        Exception: Si ocurre algún error al realizar el oversampling o undersampling.
    """
    
    global data
    if args.mode != "test":
        try:
            if args.preproceso["sampling"] == "oversampling":
                ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = ros.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN+"Oversampling realizado con éxito"+Fore.RESET)
            elif args.preproceso["sampling"] == "undersampling":
                rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x, y = rus.fit_resample(x, y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
                print(Fore.GREEN+"Undersampling realizado con éxito"+Fore.RESET)
            elif args.preproceso["sampling"] == "SMOTE":
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x,y=SMOTE().fit_resample(x,y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)
            elif args.preproceso["sampling"] == "ADASYN":
                x = data.drop(columns=[args.prediction])
                y = data[args.prediction]
                x,y=ADASYN().fit_resample(x,y)
                x = pd.DataFrame(x, columns=data.drop(columns=[args.prediction]).columns)
                y = pd.Series(y, name=args.prediction)
                data = pd.concat([x, y], axis=1)

            else:
                print(Fore.YELLOW+"No se están realizando oversampling o undersampling"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al realizar oversampling o undersampling"+Fore.RESET)
            print(e)
            sys.exit(1)
    else:
        print(Fore.GREEN+"No se realiza oversampling o undersampling en modo test"+Fore.RESET)

def drop_features():
    """
    Elimina las columnas especificadas del conjunto de datos.

    Parámetros:
    features (list): Lista de nombres de columnas a eliminar.

    """
    global data
    try:
        data = data.drop(columns=args.preprocessing["drop_features"])
        print(Fore.GREEN+"Columnas eliminadas con éxito"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al eliminar columnas"+Fore.RESET)
        print(e)
        sys.exit(1)



# Funciones para entrenar un modelo

def divide_data():
    """
    Función que divide los datos en conjuntos de entrenamiento y desarrollo.

    Parámetros:
    - data: DataFrame que contiene los datos.
    - args: Objeto que contiene los argumentos necesarios para la división de datos.

    Retorna:
    - x_train: DataFrame con las características de entrenamiento.
    - x_dev: DataFrame con las características de desarrollo para evaluar el entrenamiento.
    - y_train: Serie con las etiquetas de entrenamiento.
    - y_dev: Serie con las etiquetas de desarrollo para evaluar el entrenamiento.
    - x_dev: DataFrame con las características de desarrollo para evaluar un modelo entrenado.
    - y_dev: Serie con las etiquetas de desarrollo para evaluar un modelo entrenado.
    """
    global data
    # Nombre de la columna que quieres mover
    columna_target = args.prediction
    # Reorganizar el DataFrame para mover el target al final
    data = data[[col for col in data.columns if col != columna_target] + [columna_target]]
    #  Seleccionamos las características y la clase
    x= data.iloc[:, :-1].values # Todas las columnas menos la última
    y = data.iloc[:, -1].values # Última columna
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.25, random_state=42)
    x_dev,x_devTest, y_dev,y_devTest = train_test_split(x_dev, y_dev, test_size=0.5, random_state=42)
    print(f"Shape de X_train: {x_train.shape}")
    print(f"Shape de X_dev: {x_dev.shape}")
    print(f"Shape de X_devTest: {x_devTest.shape}")
    if args.mode=='train' :
        return x_train, x_dev, y_train, y_dev
    elif args.mode == "test":
        return x_devTest,y_devTest

def save_model(gs):
    """
    Guarda el modelo y los resultados de la búsqueda de hiperparámetros en archivos.

    Parámetros:
    - gs: objeto GridSearchCV, el cual contiene el modelo y los resultados de la búsqueda de hiperparámetros.

    Excepciones:
    - Exception: Si ocurre algún error al guardar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'wb') as file:
            pickle.dump(gs, file)
            print(Fore.CYAN + "Modelo guardado con éxito" + Fore.RESET)

        # Verificar si las métricas existen en los resultados
        mean_f1 = gs.cv_results_.get('mean_test_f1_score', [None] * len(gs.cv_results_['params']))
        mean_accuracy = gs.cv_results_.get('mean_test_accuracy', [None] * len(gs.cv_results_['params']))
        mean_custom = gs.cv_results_.get(f'mean_test_custom', [None] * len(gs.cv_results_['params']))
        mean_recall= gs.cv_results_.get(f'mean_test_recall', [None] * len(gs.cv_results_['params']))

        # Obtener y ordenar resultados si existen las métricas
        results = sorted(
            zip(gs.cv_results_['params'], mean_custom, mean_f1, mean_accuracy,mean_recall),
            key=lambda x: x[1] if x[1] is not None else -1, reverse=True
        )

        # Guardar en CSV
        with open('output/modelo.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Params', args.estimator, 'F1_Score', 'Accuracy','Recall'])
            writer.writerows(results)  # Escribir filas ordenadas

        print(Fore.GREEN + "Resultados guardados en CSV ordenados de mayor a menor" + Fore.RESET)

    except Exception as e:
        print(Fore.RED + "Error al guardar el modelo" + Fore.RESET)
        print(e)

def mostrar_resultados(gs, x_dev, y_dev):
    """
    Muestra los resultados del clasificador.

    Parámetros:
    - gs: objeto GridSearchCV, el clasificador con la búsqueda de hiperparámetros.
    - x_dev: array-like, las características del conjunto de desarrollo.
    - y_dev: array-like, las etiquetas del conjunto de desarrollo.

    Imprime en la consola los siguientes resultados:
    - Mejores parámetros encontrados por la búsqueda de hiperparámetros.
    - Mejor puntuación obtenida por el clasificador.
    - F1-score micro del clasificador en el conjunto de desarrollo.
    - F1-score macro del clasificador en el conjunto de desarrollo.
    - Informe de clasificación del clasificador en el conjunto de desarrollo.
    - Matriz de confusión del clasificador en el conjunto de desarrollo.
    """
    if args.verbose:
        print(Fore.MAGENTA+"> Mejores parametros:\n"+Fore.RESET, gs.best_params_)
        print(Fore.MAGENTA+"> Mejor puntuacion:\n"+Fore.RESET, gs.best_score_)
        print(Fore.MAGENTA+"> F1-score micro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[0])
        print(Fore.MAGENTA+"> F1-score macro:\n"+Fore.RESET, calculate_fscore(y_dev, gs.predict(x_dev))[1])
        print(Fore.MAGENTA+"> Informe de clasificación:\n"+Fore.RESET, calculate_classification_report(y_dev, gs.predict(x_dev)))
        print(Fore.MAGENTA+"> Matriz de confusión:\n"+Fore.RESET, calculate_confusion_matrix(y_dev, gs.predict(x_dev)))

def kNN():
    """
    Función para implementar el algoritmo kNN.
    Hace un barrido de hiperparámetros para encontrar los parámetros óptimos.

    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    global data

    # Dividimos los datos en entrenamiento y validación
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Definimos los criterios de evaluación
    scoring = {
        'custom': args.estimator,
        'f1_score': 'f1_weighted',
        'accuracy': 'accuracy',
        'recall':'recall_weighted'
    }
    
    with tqdm(total=100, desc='Procesando kNN', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(KNeighborsClassifier(), args.kNN, cv=5, n_jobs=args.cpu, scoring=scoring, refit='custom')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  
            pbar.update(random.random() * 2)  
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    
    execution_time = end_time - start_time
    print("Tiempo de ejecución:" + Fore.MAGENTA, execution_time, Fore.RESET + " segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

def decision_tree():
    """
    Función para implementar el algoritmo de árbol de decisión.

    :param data: Conjunto de datos para realizar la clasificación.
    :type data: pandas.DataFrame
    :return: Tupla con la clasificación de los datos.
    :rtype: tuple
    """
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()
    
    # Definimos los criterios de evaluación
    scoring = {
        'custom': args.estimator,
        'f1_score': 'f1_weighted',
        'accuracy': 'accuracy',
        'recall':'recall_weighted'
    }

    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando decision tree', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(DecisionTreeClassifier(), args.decision_tree, cv=5, n_jobs=args.cpu, scoring=scoring, refit='custom')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
    
def random_forest():
    """
    Función que entrena un modelo de Random Forest utilizando GridSearchCV para encontrar los mejores hiperparámetros.
    Divide los datos en entrenamiento y desarrollo, realiza la búsqueda de hiperparámetros, guarda el modelo entrenado
    utilizando pickle y muestra los resultados utilizando los datos de desarrollo.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    
    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # Definimos los criterios de evaluación
    scoring = {
        'custom': args.estimator,
        'f1_score': 'f1_weighted',
        'accuracy': 'accuracy',
        'recall':'recall_weighted'
    }
    
    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando random forest', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(RandomForestClassifier(), args.random_forest, cv=5, n_jobs=args.cpu, scoring=scoring, refit='custom')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15))  # Esperamos un tiempo aleatorio
            pbar.update(random.random()*2)  # Actualizamos la barra con un valor aleatorio
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)
def naive_bayes():

    # Dividimos los datos en entrenamiento y dev
    x_train, x_dev, y_train, y_dev = divide_data()

    # Definimos los criterios de evaluación
    scoring = {
        'custom': args.estimator,
        'f1_score': 'f1',
        'accuracy': 'accuracy',
        'recall':'recall'
    }

    # Hacemos un barrido de hiperparametros
    with tqdm(total=100, desc='Procesando naive bayes', unit='iter', leave=True) as pbar:
        gs = GridSearchCV(MultinomialNB(), args.naive_bayes, cv=5, n_jobs=args.cpu, scoring=scoring,refit='custom')
        start_time = time.time()
        gs.fit(x_train, y_train)
        end_time = time.time()
        for i in range(100):
            time.sleep(random.uniform(0.06, 0.15)) 
            pbar.update(random.random()*2)  
        pbar.n = 100
        pbar.last_print_n = 100
        pbar.update(0)
    execution_time = end_time - start_time
    print("Tiempo de ejecución:"+Fore.MAGENTA, execution_time,Fore.RESET+ "segundos")
    
    # Mostramos los resultados
    mostrar_resultados(gs, x_dev, y_dev)
    
    # Guardamos el modelo utilizando pickle
    save_model(gs)

# Funciones para predecir con un modelo

def load_model():
    """
    Carga el modelo desde el archivo 'output/modelo.pkl' y lo devuelve.

    Returns:
        model: El modelo cargado desde el archivo 'output/modelo.pkl'.

    Raises:
        Exception: Si ocurre un error al cargar el modelo.
    """
    try:
        with open('output/modelo.pkl', 'rb') as file:
            model = pickle.load(file)
            print(Fore.GREEN+"Modelo cargado con éxito"+Fore.RESET)
            return model
    except Exception as e:
        print(Fore.RED+"Error al cargar el modelo"+Fore.RESET)
        print(e)
        sys.exit(1)
        
def predict():
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        Ninguno

    Retorna:
        Ninguno
    """
    global data
   # Seleccionamos las características (todas las columnas menos la de predicción)
    x_data = data.drop(columns=[args.prediction])

    # Predecimos
    prediction = model.predict(x_data)
    
    prediction_column_name = args.prediction + "Prediccion"
    data = pd.concat([data, pd.DataFrame(prediction, columns=[prediction_column_name])], axis=1)
def predictTest(X_devTest):
    """
    Realiza una predicción utilizando el modelo entrenado y guarda los resultados en un archivo CSV.

    Parámetros:
        X_devTest:Datos para evaluar sin el targert
    Retorna:
        X_devTest:Datos evaaluados con el targert
    """
    global data
    if not isinstance(X_devTest, pd.DataFrame):
        X_devTest = pd.DataFrame(X_devTest, columns=data.columns[:-1])



    # Predecimos
    prediction = model.predict(X_devTest)
    
    prediction_column_name = args.prediction + "Prediccion"

    print(X_devTest.columns)  # Asegura que X_devTest tenga nombres de columnas

    X_devTest = pd.concat([X_devTest, pd.DataFrame(prediction, columns=[prediction_column_name])], axis=1)

    return X_devTest



    
# Función principal

if __name__ == "__main__":
    #Cargo el JSON
    with open('config.json', 'r', encoding='utf-8') as file:
        miJson = json.load(file)  
    # Fijamos la semilla
    np.random.seed(42)
    print("=== Clasificador ===")
    # Manejamos la señal SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    # Parseamos los argumentos
    args = parse_args()
    # Si la carpeta output no existe la creamos
    print("\n- Creando carpeta output...")
    try:
        os.makedirs('output')
        print(Fore.GREEN+"Carpeta output creada con éxito"+Fore.RESET)
    except FileExistsError:
        print(Fore.GREEN+"La carpeta output ya existe"+Fore.RESET)
    except Exception as e:
        print(Fore.RED+"Error al crear la carpeta output"+Fore.RESET)
        print(e)
        sys.exit(1)
    # Cargamos los datos
    print("\n- Cargando datos...")
    data = load_data(args.file)
    # Descargamos los recursos necesarios de nltk
    print("\n- Descargando diccionarios...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    # Preprocesamos los datos
    print("\n- Preprocesando datos...")
    preproceso()
    if args.debug:
        try:
            print("\n- Guardando datos preprocesados...")
            data.to_csv('output/data-processed.csv', index=False)
            print(Fore.GREEN+"Datos preprocesados guardados con éxito"+Fore.RESET)
        except Exception as e:
            print(Fore.RED+"Error al guardar los datos preprocesados"+Fore.RESET)
    if args.mode == "train":
        # Ejecutamos el algoritmo seleccionado
        print("\n- Ejecutando algoritmo...")
        if args.algorithm == "kNN":
            try:
                kNN()
                print(Fore.GREEN+"Algoritmo kNN ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "decision_tree":
            try:
                decision_tree()
                print(Fore.GREEN+"Algoritmo árbol de decisión ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "random_forest":
            try:
                random_forest()
                print(Fore.GREEN+"Algoritmo random forest ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)
        elif args.algorithm == "naive_bayes":
            try:
                naive_bayes()
                print(Fore.GREEN+"Algoritmo naive bayes ejecutado con éxito"+Fore.RESET)
                sys.exit(0)
            except Exception as e:
                print(e)      
        else:
            print(Fore.RED+"Algoritmo no soportado"+Fore.RESET)
            sys.exit(1)
    elif args.mode == "test":
        # Cargamos el modelo
        print("\n- Cargando modelo...")
        model = load_model()
        # Predecimos
        print("\n- Prediciendo...")
        try:
            x_devTest,y_devTest=divide_data()
            X_devTestPred=predictTest(x_devTest)
            print(Fore.GREEN+"Predicción realizada con éxito"+Fore.RESET)
            # Guardamos el dataframe con la prediccion
            #data.to_csv('output/data-prediction.csv', index=False)
            y_pred=X_devTestPred[args.prediction + "Prediccion"]
            dataf=pd.concat([X_devTestPred, pd.DataFrame(y_devTest, columns=["Target"])], axis=1)
            calcular_metricas_y_guardar(y_devTest, y_pred, dataf)
            
            # Calcula las métricas
            print(Fore.MAGENTA + "> F1-score micro:\n" + Fore.RESET, calculate_fscore(y_devTest, y_pred))
            print(Fore.MAGENTA + "> F1-score macro:\n" + Fore.RESET, calculate_fscore(y_devTest, y_pred))
            print(Fore.MAGENTA + "> Informe de clasificación:\n" + Fore.RESET, calculate_classification_report(y_devTest, y_pred))
            print(Fore.MAGENTA + "> Matriz de confusión:\n" + Fore.RESET, calculate_confusion_matrix(y_devTest, y_pred))

            print(Fore.GREEN + "Predicción guardada con éxito" + Fore.RESET)
            sys.exit(0)
        except Exception as e:
            print("adios")
            print(e)
            sys.exit(1)
    else:
        print(Fore.RED+"Modo no soportado"+Fore.RESET)
        sys.exit(1)
