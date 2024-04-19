import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PowerTransformer
import umap



def UMAP_GPT(data,clin_data,color_col,shape_col="loc",Titre=None):
    clin_data_series = clin_data[color_col]
    col_to_drop = ["__no_feature","__ambiguous","__too_low_aQual","__not_aligned","__alignment_not_unique"]
    col_to_drop += list(data.columns[data.sum(axis=0) < 100]) 
    data = data.drop(col_to_drop,axis=1)
    
    
    data = data.loc[clin_data_series.index]
    
    # data_normalized = normalize_data(data)
    # Étape de prétraitement : Standardisation des données (centrage-réduction)
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)

    # Réduction de dimension avec UMAP
    umap_model = umap.UMAP(n_components=2)  # Sélectionnez le nombre de dimensions souhaitées (ici, 2 pour une visualisation en 2D)
    umap_result = umap_model.fit_transform(data_standardized)

    # Créer un DataFrame pour stocker les résultats UMAP
    umap_df = pd.DataFrame(data=umap_result, columns=['UMAP1', 'UMAP2'], index=data.index)

    # # Visualisation des deux premières composantes principales
    # plt.scatter(umap_df['PC1'], umap_df['PC2'],c=clin_data_series.map(mappage))
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('Analyse PCA des données RNAseq')
    # plt.show()

    # Ajouter la colonne de couleurs à umap_df
    if color_col is not None:
        color_name = clin_data_series.name
        umap_df[color_name] = clin_data_series
    
    #Ajouter la colonne de shape des point
    if shape_col is not None:
        umap_df[shape_col] = clin_data.loc[umap_df.index,shape_col]
    
    
    # Créer un plot interactif avec Plotly Express
    fig = px.scatter(umap_df, x='UMAP1', y='UMAP2', color=color_name, hover_name=umap_df.index,symbol=shape_col,title=Titre)

    # Afficher le plot interactif
    fig.show()
    return(umap_result,fig)


def PCA_GPT(data,clin_data,color_col,shape_col="loc",Titre=None):
    clin_data_series = clin_data[color_col]
    col_to_drop = ["__no_feature","__ambiguous","__too_low_aQual","__not_aligned","__alignment_not_unique"]
    col_to_drop += list(data.columns[data.sum(axis=0) < 100]) 
    data = data.drop(col_to_drop,axis=1)
    
    
    data = data.loc[clin_data_series.index]
    
    # data_normalized = normalize_data(data)
    # Étape de prétraitement : Standardisation des données (centrage-réduction)
    scaler = StandardScaler()

    data_standardized = scaler.fit_transform(data)

    # Appliquer la PCA
    pca = PCA(n_components=2)  # Sélectionnez le nombre de composantes principales souhaitées (ici, 2 pour visualiser en 2D)
    pca_result = pca.fit_transform(data_standardized)


    # Créer un DataFrame pour stocker les résultats de PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=data.index)

    # # Visualisation des deux premières composantes principales
    # plt.scatter(pca_df['PC1'], pca_df['PC2'],c=clin_data_series.map(mappage))
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('Analyse PCA des données RNAseq')
    # plt.show()

    # Ajouter la colonne de couleurs à pca_df
    if color_col is not None:
        color_name = clin_data_series.name
        pca_df[color_name] = clin_data_series
    
    #Ajouter la colonne de shape des point
    if shape_col is not None:
        pca_df[shape_col] = clin_data.loc[pca_df.index,shape_col]
    
    
    # Créer un plot interactif avec Plotly Express
    fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_name, hover_name=pca_df.index,symbol=shape_col,title=Titre)

    # Afficher le plot interactif
    fig.show()
    return(pca,fig)


def PCA_GPT_Power(data,clin_data,color_col,shape_col="loc",Titre=None):
    """
    Réalise une Analyse en Composantes Principales (PCA) pour visualiser les données en deux dimensions.

    Parameters:
        data (DataFrame): Les données à analyser.
        clin_data (DataFrame): Les données cliniques associées pour la coloration.
        color_col (str): Le nom de la colonne dans 'clin_data' à utiliser pour la coloration.
        shape_col (str, optional): Le nom de la colonne dans 'clin_data' à utiliser pour la forme des points. Par défaut, "loc".
        Titre (str, optional): Le titre du graphique. Par défaut, None.

    Returns:
        tuple: Un tuple contenant l'objet PCA et le graphique interactif Plotly Express.

    Raises:
        ValueError: Si les colonnes spécifiées (color_col, shape_col) ne sont pas présentes dans 'clin_data'.
    """
    clin_data_series = clin_data[color_col]
    # col_to_drop = ["__no_feature","__ambiguous","__too_low_aQual","__not_aligned","__alignment_not_unique"]
    col_to_drop = list(data.columns[data.sum(axis=0) < 100]) 
    data = data.drop(col_to_drop,axis=1)
    
    
    data = data.loc[clin_data_series.index]
    
    # data_normalized = normalize_data(data)
    # Étape de prétraitement : Standardisation des données (centrage-réduction)
    scaler = PowerTransformer()

    data_standardized = scaler.fit_transform(data)

    # Appliquer la PCA
    pca = PCA(n_components=2)  # Sélectionnez le nombre de composantes principales souhaitées (ici, 2 pour visualiser en 2D)
    pca_result = pca.fit_transform(data_standardized)


    # Créer un DataFrame pour stocker les résultats de PCA
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=data.index)

    # # Visualisation des deux premières composantes principales
    # plt.scatter(pca_df['PC1'], pca_df['PC2'],c=clin_data_series.map(mappage))
    # plt.xlabel('PC1')
    # plt.ylabel('PC2')
    # plt.title('Analyse PCA des données RNAseq')
    # plt.show()

    # Ajouter la colonne de couleurs à pca_df
    if color_col is not None:
        color_name = clin_data_series.name
        pca_df[color_name] = clin_data_series
    
    #Ajouter la colonne de shape des point
    if shape_col is not None:
        pca_df[shape_col] = clin_data.loc[pca_df.index,shape_col]
    
    
    # Créer un plot interactif avec Plotly Express
    fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_name, hover_name=pca_df.index,symbol=shape_col,title=Titre)

    # Afficher le plot interactif
    fig.show(renderer='iframe')
    return(pca,fig)