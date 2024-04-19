import pandas as pd
import tqdm
import numpy as np
from pyliftover import LiftOver

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
