from pybiomart import Server
import pandas as pd


def convert_symbols_to_human(ranked_mouse, m2h=None, column_name="Symbol"):
    if m2h is None:
        # Se connecter au serveur Biomart
        server = Server(host='http://www.ensembl.org')

        # Sélectionner le dataset pour les gènes de souris
        dataset = (server.marts['ENSEMBL_MART_ENSEMBL']
                         .datasets['mmusculus_gene_ensembl'])

        # Effectuer la requête pour la conversion des symboles de souris en symboles humains
        m2h = dataset.query(attributes=['external_gene_name',
                                        'hsapiens_homolog_associated_gene_name'])

    # Effectuer la conversion en utilisant la méthode map()
    ranked_mouse["Symbol humain"] = ranked_mouse[column_name].map(
        m2h.drop_duplicates('Gene name').set_index('Gene name')['Human gene name']
    )

    # Créer le DataFrame "ranked_human" en enlevant les doublons et en utilisant "Symbol humain" comme index
    ranked_human = ranked_mouse.drop_duplicates('Symbol humain').set_index("Symbol humain")

    return ranked_human


def gsea_Kegg_Mousegenatla(ranked,outDir=None):
    if outDir is not None:
        os.makedirs(outDir,exist_ok=True)
    pre_res_mouse_gene_atlas = gp.prerank(rnk = ranked,
                         gene_sets = "Mouse_Gene_Atlas",
                                          threads=4,
                                          outDir=os.path.join(outDir,"GSEA_Mouse_Gene_Atlas"),
                         seed = 6, permutation_num = 1000)

    pre_res_kegg = gp.prerank(rnk = ranked,
                         gene_sets = "KEGG_2019_Mouse",
                                          threads=4,
                                          outDir=os.path.join(outDir,"GSEA_Kegg_2019_Mouse"),
                         seed = 6, permutation_num = 1000)

    return pre_res_mouse_gene_atlas,pre_res_kegg