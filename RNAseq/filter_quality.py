# Compter le nombre de gènes avec au moins 1 lecture pour chaque échantillon
import plotly.express as px

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


