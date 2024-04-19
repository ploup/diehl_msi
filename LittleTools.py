#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def replace_index_pattern(df, pattern, replace_pattern, regex=True, axis=0):
    """
    Remplace un motif dans les index ou les colonnes d'un DataFrame Pandas.

    Parameters:
        df (DataFrame): Le DataFrame contenant les index ou les colonnes à modifier.
        pattern (str): Le motif à rechercher dans les index ou les colonnes.
        replace_pattern (str): Le motif de remplacement pour les index ou les colonnes.
        regex (bool, optional): Si True, utilise les expressions régulières pour la recherche et le remplacement. Par défaut, True.
        axis (int, optional): L'axe le long duquel effectuer le remplacement : 0 pour les index, 1 pour les colonnes. Par défaut, 0.

    Returns:
        DataFrame: Le DataFrame avec les index ou les colonnes modifiés.

    Raises:
        ValueError: Si l'axe spécifié n'est ni 0 (pour les index) ni 1 (pour les colonnes).
    """
    if axis == 0:
        df.index = df.index.str.replace(pattern, replace_pattern, regex=regex)
    elif axis == 1:
        df.columns = df.columns.str.replace(pattern, replace_pattern, regex=regex)
    else:
        raise ValueError("Axis must be 0 (for index) or 1 (for columns)")
    
    return df
