import pandas as pd
import tqdm
import numpy as np
from pyliftover import LiftOver
import os
#%%
def indel_len_profil(files,nb_read_lim=20):
    """
    Sort un table avec pour chaque ligne un patient et pour chaque colonne un MS.
    Les valeur correspondent à la taille en base de l'indel, et np.nan si pas d'info
    
    
    Parameters
    ----------
    files : list(nb_patient)
        list of res files path.
    nb_read_lim : int, optional
        Minimal depth in Tumoral and Normal. The default is 10.
    
    Returns
    -------
    pd.DataFrame
        Dataframe of mutation with patients as rows and MS as columns. 0 means non-mutated, 1 means mutated
    
    """
    df_mut_map = pd.DataFrame()
    
    for path_res in tqdm.tqdm(files):
    
        df = pd.read_table(path_res,sep='\t',index_col=0)
        samp_name = path_res.split('/')[-1][:-4]
        
        # filtre les MS faiblement couvert
        df = df[(df['sum_Tum']>=nb_read_lim) & (df['sum_Nor']>=nb_read_lim)]
        
        # replace NaN par 0 (non muté)
        df['del_ins_size'] = df['del_ins_size'].replace(np.NaN,0)

        df_mut_map = df_mut_map.append(pd.Series(df['del_ins_size'].T,name=samp_name,dtype=int))
        
            
    return df_mut_map


 

#%%
# def clin_gen_mutated_row(files, delthreshold=10, nb_read_lim=20):
#     """
#     Sort un table avec pour chaque ligne un patient et pour chaque colonne un MS. Un 0 si non muté chez le patient, 1 si muté, et np.nan si pas d'info
    
    
#     Parameters
#     ----------
#     files : list(nb_patient)
#         list of res files path.
#     nb_read_lim : int, optional
#         Minimal depth in Tumoral and Normal. The default is 10.
    
#     Returns
#     -------
#     pd.DataFrame
#         Dataframe of mutation with patients as rows and MS as columns. 0 means non-mutated, 1 means mutated
    
#     """
#     df_mut_map = pd.DataFrame()
    
#     for path_res in tqdm.tqdm(files):
    
#         df = pd.read_table(path_res,sep='\t',index_col=0)
#         samp_name = path_res.split('/')[-1][:-4]
#         df = df[(df['sum_Tum']>=nb_read_lim) & (df['sum_Nor']>=nb_read_lim)]
#         df['Mutated'] = 0
#         mutated_index = df[(df[['-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','1','2']] > delthreshold).any(axis=1)].index
#         df.loc[mutated_index,'Mutated'] = 1
#         df_mut_map = df_mut_map.append(pd.Series(df['Mutated'].T,name=samp_name,dtype=int))
        
            
#     return df_mut_map

import pandas as pd
import numpy as np

def Generate_mutation_deletion_deltaratiosum(files, delthreshold=10, nb_read_lim=20, include_insertion=False):
    """
    Génère un tableau où chaque ligne représente un patient et chaque colonne un MS. 
    La valeur est 0 si le patient n'est pas muté, 1 s'il est muté, et np.nan s'il n'y a pas d'information.

    Parameters
    ----------
    files : list
        Liste des chemins des fichiers de résultats.
    delthreshold : int, optional
        Seuil minimal de profondeur en tumoral et normal. Par défaut, 10.
    nb_read_lim : int, optional
        Profondeur minimale en Tumoral et Normal. Par défaut, 20.
    include_insertion : bool, optional
        Indique si les insertions doivent être incluses dans l'analyse. Par défaut, False.

    Returns
    -------
    pd.DataFrame, pd.DataFrame, pd.DataFrame
        Trois DataFrames : 
        - df_mut_map : Tableau de mutations avec les patients en lignes et les MS en colonnes. 0 signifie non muté, 1 signifie muté.
        - df_del_map : Tableau de tailles de délétions avec les patients en lignes et les MS en colonnes.
        - df_delta_map : Tableau des sommes des ratios delta avec les patients en lignes et les MS en colonnes.
    """

    mutation_data = []  # Stocke les données de mutation
    delta_ratio_data = []  # Stocke les données de somme des ratios delta
    deletion_size_data = []  # Stocke les données de taille de délétion
    delta_ratio_data_del = []
    delta_ratio_data_ins = []
    if include_insertion:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2'] # Estimation de VAF cf Cody
    else:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1'] # VAF de deletion 

    deletion_col = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1']
    insertion_col = ['1','2']

    for path_res in files:
        df = pd.read_table(path_res, sep='\t', index_col=0)
        samp_name = path_res.split('/')[-1][:-4]  # Nom de l'échantillon
        df = df[(df['sum_Tum'] >= nb_read_lim) & (df['sum_Nor'] >= nb_read_lim)]  # Filtrer par seuil de profondeur

        df['Mutated'] = (df[indel_len] > delthreshold).any(axis=1)  # Calculer la colonne 'Mutated'
        df['deltaratio_sum_del'] = df[deletion_col][df[deletion_col] > 0].sum(axis=1)  # Calculer la somme des ratios delta de deletion
        df['deltaratio_sum_ins'] = df[insertion_col][df[insertion_col] > 0].sum(axis=1)  # Calculer la somme des ratios delta de insertion

        delta_ratio_data_del.append(df['deltaratio_sum_del'].rename(samp_name))
        delta_ratio_data_ins.append(df['deltaratio_sum_ins'].rename(samp_name))
        delta_ratio_data.append((df['deltaratio_sum_ins']+df['deltaratio_sum_del']).rename(samp_name))

        deletion_size_data.append(df["del_ins_size"].fillna(0).astype(int).rename(samp_name))
        mutation_data.append(df['Mutated'].astype(int).rename(samp_name))
        

    df_mut_map = pd.DataFrame(mutation_data)  # Créer le DataFrame de mutations
    df_del_map = pd.DataFrame(deletion_size_data)  # Créer le DataFrame de tailles de délétions
    df_delta_map = pd.DataFrame(delta_ratio_data)  # Créer le DataFrame des sommes des ratios delta
    df_delta_map_del = pd.DataFrame(delta_ratio_data_del) # Créer le DataFrame des sommes des ratios delta deletion
    df_delta_map_ins = pd.DataFrame(delta_ratio_data_ins) # Créer le DataFrame des sommes des ratios delta insertions

    return df_mut_map, df_del_map, df_delta_map, df_delta_map_del , df_delta_map_ins

def Generate_mutation_per_gene(files, delthreshold=10, nb_read_lim=20, include_insertion=False, coding=False):
    """
    Generate mutation per gene based on input files and parameters.

    Args:
        files (list): List of input files
        delthreshold (int, optional): Deletion threshold. Defaults to 10.
        nb_read_lim (int, optional): Read limit. Defaults to 20.
        include_insertion (bool, optional): Flag to include insertion. Defaults to False.
        coding (bool, optional): Flag to filter by coding regions. Defaults to False.

    Returns:
        pandas.DataFrame: DataFrame containing mutation data per gene
    """

    mutation_data = []  # Stocke les données de mutation

    if include_insertion:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2'] # Estimation de VAF cf Cody
    else:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1'] # VAF de deletion 


    for path_res in files:
        df = pd.read_table(path_res, sep='\t', index_col=0)
        samp_name = path_res.split('/')[-1][:-4]  # Nom de l'échantillon
        df = df[(df['sum_Tum'] >= nb_read_lim) & (df['sum_Nor'] >= nb_read_lim)]  # Filtrer par seuil de profondeur

        df['Mutated'] = (df[indel_len] > delthreshold).any(axis=1)  # Calculer la colonne 'Mutated'



        if coding:
            df = df[df['Func.refGene'].isin(['exonic', 'exonic;splicing', 'splicing',"ncRNA_exonic","ncRNA_exonic;splicing"])]

  


        mutation_data.append(df.groupby('Gene.refGene')['Mutated'].sum().rename(samp_name))

    df_mut_map = pd.DataFrame(mutation_data)  # Créer le DataFrame de mutations


    return df_mut_map

def pic_majoritaire(df):
    # Find the index of the first occurrence of the minimum value in each row
    idx_max = df[['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1',"1","2"]].idxmax(axis=1)



    # If you want to get the index of the column, not the value, you can do:
    df['pic_indel_majo'] = idx_max
    df.loc[(df["del_ins_size"].isna()),'pic_indel_majo'] = np.nan
    return df

def indel_pic_majo(files, nb_read_lim=20):


    deletion_size_data = []  # Stocke les données de taille de délétion

    for path_res in files:
        df = pd.read_table(path_res, sep='\t', index_col=0)
        df = pic_majoritaire(df)

        samp_name = path_res.split('/')[-1][:-4]  # Nom de l'échantillon
        df = df[(df['sum_Tum'] >= nb_read_lim) & (df['sum_Nor'] >= nb_read_lim)]  # Filtrer par seuil de profondeur


        deletion_size_data.append(df["pic_indel_majo"].fillna(0).astype(int).rename(samp_name))

    df_del_map = pd.DataFrame(deletion_size_data)  # Créer le DataFrame de tailles de délétions

    return  df_del_map

def clin_gen_mutated_row(files, delthreshold=10, nb_read_lim=20,include_insertion=False):
    """
    Sort un table avec pour chaque ligne un patient et pour chaque colonne un MS. Un 0 si non muté chez le patient, 1 si muté, et np.nan si pas d'info
    
    
    Parameters
    ----------
    files : list(nb_patient)
        list of res files path.
    nb_read_lim : int, optional
        Minimal depth in Tumoral and Normal. The default is 10.
    
    Returns
    -------
    pd.DataFrame
        Dataframe of mutation with patients as rows and MS as columns. 0 means non-mutated, 1 means mutated
    
    """
    mutation_data = []
    if include_insertion:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2']

    else : 
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1']

    
    for path_res in files:
        df = pd.read_table(path_res, sep='\t', index_col=0)
        samp_name = path_res.split('/')[-1][:-4]
        df = df[(df['sum_Tum'] >= nb_read_lim) & (df['sum_Nor'] >= nb_read_lim)]
        df['Mutated'] = (df[indel_len] > delthreshold).any(axis=1)
        mutation_data.append(df['Mutated'].astype(int).rename(samp_name))
    
    df_mut_map = pd.DataFrame(mutation_data)
    
    return df_mut_map


def filtnan(tabmut,NAmax=0.05): 
    """
    filter out MS with too much NA

    Parameters
    ----------
    tabmut : pd.DataFrame, shape(nb_patient,nb_MS)
        DataFrame with mutation profile.
    NAmax : float() [0,1], optional
        Maximum rate of NA in a column (MS). The default is 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame with mutation profile without columns with too much NA.

    """
    #drop les colonnes ayant moins de nb_measure_mini
    narate = tabmut.isna().sum()/tabmut.shape[0]
    return tabmut.loc[:,narate<NAmax]


def lift_on_df(df,chridcol,fromdb,todb):
    """ Fonction pour traduire d'une version Hg à une autre en utilisant LiftOver.
    Sort le df traduit accompagné des lignes dupliqué (deux position dans une version peuvent etre placé à la même position dans l'autre version)
    /!\ Ne tournera pas sur un DataFrame déjà traduit
    return df,df_dup,leftover
    """
    if chridcol == 'index':
        test=pd.DataFrame(df.index.str.split('.').to_list())[[0,1]]
        test.index = df.index

        df[['chr','loc']] = test
    else:        
        test=pd.DataFrame(df[chridcol].str.split('.').to_list())[[0,1]]
        test.index = df.index
        df[['chr','loc']] = test
    fomdbchridcol = fromdb+"chrid"
    todbchridcol = todb+'chrid'
    
    lo = LiftOver(fromdb, todb)
    df = df.astype({'loc':'int64'})
    df[todbchridcol] = str()
    leftover = []
    
    for ind,row in tqdm.tqdm(df.iterrows(),total=df.shape[0]):
        conv = lo.convert_coordinate(row['chr'],row['loc'])
        try:
            len(conv)
        except TypeError:
            leftover.append([ind,'Error'])
            continue
        if len(conv) == 1:
            df.at[ind,todbchridcol] = "{}.{}".format(conv[0][0],str(conv[0][1]))
        elif len(conv) ==0:
            leftover.append([ind,'NoConv'])
        elif len(conv) > 1:
            df.at[ind,todbchridcol] =  "{}.{}".format(conv[0][0],str(conv[0][1]))

     
    #df = df.drop(['chr','loc'],axis=1)    
    
    df[fomdbchridcol] = df.index
    df = df.set_index(todbchridcol)

    
    # keep track of duplicated index
    df_dup = df[df.index.duplicated(keep='first')] 
    # drop duplicates   
    df= df[~df.index.duplicated(keep='first')]   
    
    return df


#%%

def Vaf_profile_old_format(files):
    """
    Generates a VAF profile from the given list of files.

    Args:
        files: A list of file paths.

    Returns:
        DataFrame: A DataFrame containing the VAF profile.
    """

    df_vaf_map = pd.DataFrame()
    
    for path in tqdm.tqdm(files):
    
        df = pd.read_table(path,sep=' ',index_col=1,header=None)
        base_name = os.path.basename(path)
        while True:
            base, ext = os.path.splitext(base_name)
            if ext == '':
                break
            base_name = base
        samp_name = base_name        


        df_vaf_map = pd.concat([df_vaf_map,df[2].rename(samp_name)],axis=1)
        
            
    return df_vaf_map.T


def Vaf_profile(files):
    """
    Generates a VAF profile from the given list of files.

    Args:
        files: A list of file paths.

    Returns:
        DataFrame: A DataFrame containing the VAF profile.
    """

    df_vaf_map = pd.DataFrame()
    
    for path in tqdm.tqdm(files):
    
        df = pd.read_table(path,sep=' ',index_col=1,header=None)
        base_name = os.path.basename(path)
        while True:
            base, ext = os.path.splitext(base_name)
            if ext == '':
                break
            base_name = base
        samp_name = base_name        


        df_vaf_map = pd.concat([df_vaf_map,df[4].rename(samp_name)],axis=1)
        
            
    return df_vaf_map.T