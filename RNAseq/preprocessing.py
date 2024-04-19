# Compter le nombre de gènes avec au moins 1 lecture pour chaque échantillon
import plotly.express as px
import pandas as pd
from pybiomart import Dataset
import tqdm
import numpy as np
#%%
def TPM_merged_table(files):
    """
    Sort un table avec pour chaque colonne un patient et pour chaque ligne un gene.
    Les valeurs correspondent au TPM
    
    
    Parameters
    ----------
    files : list(nb_patient)
        list of res files path.
    
    Returns
    -------
    pd.DataFrame
        Dataframe of mutation with patients as rows and MS as columns. 0 means non-mutated, 1 means mutated
    
    """
    df_TPM_map = pd.DataFrame()
    
    for path_res in tqdm.tqdm(files):
    
        df = pd.read_table(path_res,sep='\t',index_col=0)
        samp_name = path_res.split('/')[-1].split('_')[0]
        


        df_TPM_map = df_TPM_map.append(pd.Series(df['TPM'].T,name=samp_name,dtype=int))
        
            
    return df_TPM_map



def compter_genes_avec_lecture(row):
    return (row >= 1).sum()


def compter_nb_gene_par_sample(df,axis=1):
    # Appliquer la fonction compter_genes_avec_lecture à chaque ligne (échantillon)
     return df.apply(compter_genes_avec_lecture, axis=axis)

def barplot_nb_gene(clin_data, y_col, color_col, hline_height=16000):
    """
    Crée un diagramme en barres représentant le nombre de gènes en fonction d'une colonne spécifiée, avec une ligne horizontale à y=16000.

    Parameters:
        clin_data (DataFrame): Le DataFrame contenant les données à visualiser.
        y_col (str): Le nom de la colonne contenant les données pour l'axe y (nombre de gènes).
        color_col (str): Le nom de la colonne pour colorer les barres en fonction de cette colonne.
        hline_height (int, optional): La hauteur de la ligne horizontale à ajouter. Par défaut, 16000.

    Returns:
        None: Affiche le diagramme en barres interactif.

    Raises:
        PlotlyError: Si une erreur se produit lors de la création du graphique interactif.
    """
    # Créer le diagramme en barres
    fig = px.bar(clin_data, y=y_col, color=color_col,)
    
    # Organiser les barres sur l'axe x en fonction du nombre de gènes
    fig.update_layout(
        xaxis=dict(
            categoryorder='total ascending'
        )
    )
    
    # Ajout de la ligne horizontale à y=16000
    fig.add_hline(y=hline_height, line_dash="dash", line_color="red", annotation_text=f"nb_gene={hline_height}", annotation_position="bottom right")
    
    # Ajuster la hauteur du layout
    fig.layout.height = 500
    
    # Afficher le diagramme
    fig.show()
    return fig


def cut_nb_gene_deseq(counts_df,smallestGroupSize=3,readthr=10,drop__col=0):

    
    # Identify the columns to drop from counts_df based on sum thresholds and non gene col
    if drop__col:
        col_to_drop = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]
    
        counts_df = counts_df.drop(col_to_drop, axis=1)

    gene_to_keep = counts_df.apply(lambda row: (row >= 10).sum(), axis=0) >= smallestGroupSize
    counts_df = counts_df.loc[:,gene_to_keep]

    return counts_df


def convert_ensg_to_gene_id(df, axis=1):
    """
    Convertit les IDs Ensemble (ENSG) en Gene ID dans un DataFrame.

    Cette fonction utilise Biomart pour convertir les IDs Ensemble (ENSG) en Gene ID
    et renomme les colonnes ou les index du DataFrame en conséquence.

    Parameters:
        df (DataFrame): Le DataFrame contenant les IDs à convertir.
        axis (int, optional): L'axe à convertir (1 pour colonnes, 0 pour index). Par défaut, 1.

    Returns:
        DataFrame: Le DataFrame avec les IDs convertis.
    """
    # Connexion à Biomart
    dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')

    # Récupérer la conversion des IDs
    response = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    response['Gene name'] = response['Gene name'].fillna(response['Gene stable ID'])
    
    # Créer un dictionnaire de correspondance entre les IDs
    id_mapping = {row['Gene stable ID']: row['Gene name'] for _, row in response.iterrows()}



    # Convertir les IDs
    if axis == 1:
        df = df.T
    
    df = df.rename(index=id_mapping)

    # Suppression des lignes dont l'index commence par 'ENSG' (qui n'ont pas de correspondance gene_name)
    df = df[~df.index.str.startswith('ENSG')]

    # Identifier les index dupliqués
    duplicated_index = df.index[df.index.duplicated()]

    # Si il y a des index duplqué, on les moyenne
    if len(duplicated_index) > 0:
        df = df.groupby(level=0).mean()

    if axis == 1:
        return df.T
    else:
        return df
