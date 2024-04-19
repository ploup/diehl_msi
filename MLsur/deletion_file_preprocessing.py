import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.preprocessing import OneHotEncoder

def seuil_to_binary(value, seuil):
    """
    Transforme une valeur en 1 si elle est supérieure au seuil, en 0 sinon.
    Si la valeur d'origine est NaN, elle retourne NaN.

    Parameters:
        value (float): La valeur d'origine à transformer.
        seuil (float): La valeur seuil de comparaison.

    Returns:
        float: 1 si value > seuil, 0 sinon, ou NaN si value est NaN.
    """
    if np.isnan(value):
        return np.nan
    else:
        return 1 if value > seuil else 0

def apply_seuil_to_binary(df, seuil):
    """
    Applique la fonction seuil_to_binary à chaque élément du DataFrame.

    Parameters:
        df (DataFrame): Le DataFrame contenant les valeurs à transformer.
        seuil (float): La valeur seuil de comparaison.

    Returns:
        DataFrame: Un nouveau DataFrame avec les valeurs transformées.
    """
    return df.applymap(lambda x: seuil_to_binary(x, seuil))

def split_dataframe(df, num_chunks):
    """
    Split a DataFrame into multiple chunks along the columns axis.

    Parameters:
        df (DataFrame): The DataFrame to be split.
        num_chunks (int): The number of chunks to split the DataFrame into.

    Returns:
        list: A list of DataFrames, each representing a chunk of the original DataFrame.
    """
    total_columns = df.shape[1]  # Nombre total de colonnes dans le DataFrame
    chunk_size = total_columns // num_chunks  # Taille approximative de chaque chunk

    # Calcul des indices de découpe
    split_indices = [i * chunk_size for i in range(1, num_chunks)]

    # Division du tableau en chunks sur l'axe des colonnes
    chunks = np.array_split(df, split_indices, axis=1)
    
    return chunks
    
def onehotencoding(df):
    """
    Generate one-hot encoding for a given DataFrame.

    Parameters:
    - df: pandas DataFrame
        The input DataFrame to be encoded.

    Returns:
    - encoded_df: pandas DataFrame
        The DataFrame with one-hot encoded columns.

    This function takes a pandas DataFrame as input and uses the `pd.get_dummies`
    function to generate one-hot encoded columns for each column in the DataFrame.
    If a column has a suffix '_nan', it is treated as a special case and the
    corresponding NaN values are encoded as well. The function then concatenates
    all the encoded columns and returns the resulting DataFrame.
    """
    return pd.concat([
          pd.get_dummies(df[col], prefix=col, dummy_na=True).where(~pd.get_dummies(df[col], prefix=col, dummy_na=True)[col+'_nan'], np.nan).drop(col+'_nan',axis=1)
          if col+'_nan' in pd.get_dummies(df[col], prefix=col, dummy_na=True).columns
          else pd.get_dummies(df[col], prefix=col, dummy_na=True)
          for col in df.columns
      ],axis=1).astype(np.float64)

from multiprocessing import Pool

def parallel_encode(df, encode_function, num_processes=4):
    """
    Multiprocess a given encoding function:
    
    Parameters:
    - df: pandas.DataFrame
        The DataFrame to be encoded.
    - encode_function: Callable
        The function used to encode each subpart of the DataFrame.
    - num_processes: int
         The number of processes to use for parallel encoding. Default is 4.
    Returns: 
    - df_encoded : pandas.DataFrame
        The encoded DataFrame.
    """
    # Divisez le DataFrame en sous-parties
    chunks = split_dataframe(df, num_processes)

    # Créez un pool de processus
    with Pool(num_processes) as pool:
        results = pool.map(encode_function, chunks)

    # Concaténez les résultats en un seul DataFrame
    df_encoded = pd.concat(results, axis=1)
    
    return df_encoded


def OneHotEncoder_withNan(df):
    """
    One-hot encodes a DataFrame column with support for missing values.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.

    Returns:
        pandas.DataFrame: The DataFrame with the one-hot encoded columns.
    """
    # Supposons que df soit votre DataFrame et col soit la colonne catégorielle que vous souhaitez décomposer
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='error')
    binary_data = encoder.fit_transform(df)
    # Créez un DataFrame avec les colonnes binaires
    binary_df = pd.DataFrame(binary_data, columns=encoder.get_feature_names_out(), index=df.index)


    prefixes = binary_df.filter(like="_nan").columns
    # Parcourez chaque préfixe

    for prefix in prefixes:
        # Sélectionnez les colonnes correspondantes pour ce préfixe, y compris la colonne avec "_nan"
        cols_for_prefix = binary_df.columns[binary_df.columns.str.startswith(prefix[:-4])]

        # Remplacez les valeurs par NaN lorsque la colonne "_nan" est égale à 1
        binary_df.loc[binary_df[prefix] == 1, cols_for_prefix] = np.nan
    binary_df = binary_df.drop(prefixes,axis=1)
    return binary_df



def OneHotEncoder_withNan_chunked_process(df, num_processes=4):
    """
    Function to process a DataFrame using the OneHotEncoder_withNan function in parallel.

    Args:
        df (pandas.DataFrame): The DataFrame to be processed.
        num_processes (int, optional): The number of processes to use for parallel processing. 
            Defaults to 4.

    Returns:
        pandas.DataFrame: The processed DataFrame after concatenating the results 
            from all the processes.
    """
    
    
    # Divisez le DataFrame en sous-parties
    chunks = np.array_split(df, num_processes)
    
    # Créez un pool de processus
    with Pool(num_processes) as pool:
        results = pool.map(OneHotEncoder_withNan, chunks)
    
    # Fusionnez les résultats en un seul DataFrame
    return pd.concat(results)