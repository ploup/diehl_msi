import pandas as pd
from usefull_msi_script.diehl_msi.MLsur.GCV_custompipelin_function import *
from sksurv.datasets import get_x_y
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import ConvergenceError, ConvergenceWarning
import warnings
from multiprocessing import Pool, cpu_count
from matplotlib import pyplot as plt
from matplotlib_venn import venn3
import os

def cox_on_col(ll, surv_tab, duration_col='os', event_col='os_event'):
    """
    Effectue une analyse de régression de Cox pour une colonne spécifique.

    Cette fonction utilise la régression de Cox pour estimer la p-valeur associée
    à la colonne fournie dans le DataFrame.

    Parameters:
        ll (DataFrame): Le DataFrame contenant les données à analyser.
        surv_tab (DataFrame): Le DataFrame contenant les données de survie.
        duration_col (str, optional): Le nom de la colonne représentant la durée de survie. Par défaut, 'os'.
        event_col (str, optional): Le nom de la colonne représentant l'événement de survie. Par défaut, 'os_event'.

    Returns:
        float: La p-valeur associée à la colonne.

    Raises:
        ConvergenceWarning: Si une convergence incomplète est détectée lors de l'ajustement du modèle.
        ConvergenceError: Si une erreur de convergence se produit lors de l'ajustement du modèle.
    """
    cox_lfl = CoxPHFitter()
    
    # Ignorer les avertissements ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            cox_lfl.fit(pd.merge(ll, surv_tab, left_index=True, right_index=True).dropna(), duration_col=duration_col, event_col=event_col)
            return cox_lfl.summary['p'].values[0]
        except ConvergenceWarning:
            return np.nan
        except ConvergenceError:
            return np.nan



def cox_on_col_withcindex(ll, surv_tab, duration_col='os', event_col='os_event'):
    """
    Calcule la p-valeur et le c-index en utilisant une analyse de survie de Cox pour une colonne donnée.

    Parameters:
        ll (Series): La colonne du DataFrame pour laquelle calculer la p-valeur et le c-index.
        surv_tab (DataFrame): Le DataFrame contenant les données de survie.
        duration_col (str, optional): Le nom de la colonne représentant la durée de l'événement. Par défaut, 'os'.
        event_col (str, optional): Le nom de la colonne représentant l'événement (0 pour non, 1 pour oui). Par défaut, 'os_event'.

    Returns:
        tuple: Un tuple contenant la p-valeur et le c-index.
            - pvalue (float): La p-valeur calculée.
            - cindex (float): Le c-index (concordance index) calculé.
    """
    cox_lfl = CoxPHFitter()
    # Ignorer les avertissements ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        try:
            cox_lfl.fit(pd.merge(ll, surv_tab, left_index=True, right_index=True).dropna(), duration_col=duration_col, event_col=event_col)
            return cox_lfl.summary['p'].values[0],cox_lfl.concordance_index_
        except ConvergenceWarning:
            return np.nan, np.nan
        except ConvergenceError:
            return np.nan, np.nan



def cox_on_col_all(ll, surv_tab,cox_lfl , duration_col='os', event_col='os_event'):
    """
    Calcule la p-valeur et le c-index en utilisant une analyse de survie de Cox pour une colonne donnée.

    Parameters:
        ll (Series): La colonne du DataFrame pour laquelle calculer la p-valeur et le c-index.
        surv_tab (DataFrame): Le DataFrame contenant les données de survie.
        duration_col (str, optional): Le nom de la colonne représentant la durée de l'événement. Par défaut, 'os'.
        event_col (str, optional): Le nom de la colonne représentant l'événement (0 pour non, 1 pour oui). Par défaut, 'os_event'.

    Returns:
        tuple: Un tuple contenant la p-valeur et le c-index.
            - pvalue (float): La p-valeur calculée.
            - cindex (float): Le c-index (concordance index) calculé.
    """
    # Ignorer les avertissements ConvergenceWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            cox_lfl.fit(pd.merge(ll, surv_tab, left_index=True, right_index=True).dropna(), duration_col=duration_col, event_col=event_col)
            df_res = cox_lfl.summary
            df_res.loc[ll.name,'cindex']= cox_lfl.concordance_index_
            return df_res
        except ConvergenceWarning:
            cox_lfl.fit(pd.merge(ll, surv_tab, left_index=True, right_index=True).dropna(), duration_col=duration_col, event_col=event_col)
            df_res = cox_lfl.summary
            df_res.loc[ll.name+"ConvergenceWarning",'cindex']= cox_lfl.concordance_index_
            return df_res
        except ConvergenceError:
            return pd.DataFrame(np.full((1,11),np.nan),columns=['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
       'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)',
       'cindex'],index=[ll.name]).rename_axis('covariate')
        except ValueError:
            return pd.DataFrame(np.full((1,11),np.nan),columns=['coef', 'exp(coef)', 'se(coef)', 'coef lower 95%', 'coef upper 95%',
       'exp(coef) lower 95%', 'exp(coef) upper 95%', 'z', 'p', '-log2(p)',
       'cindex'],index=[ll.name]).rename_axis('covariate')

def cox_on_all_itterrows(df,df_surv,duration_col='pfs',event_col="pfs_event"):
    """
    Perform Cox Proportional Hazards analysis on each column of a DataFrame iteratively.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the columns to iterate over.
        df_surv (pandas.DataFrame): The survival data DataFrame containing the duration and event columns.
        duration_col (str, optional): The name of the duration column in df_surv. Defaults to 'pfs'.
        event_col (str, optional): The name of the event column in df_surv. Defaults to 'pfs_event'.
    
    Returns:
        pandas.DataFrame: The concatenated results of CoxPHFitter analysis for each column.
    """
    cox_lfl = CoxPHFitter()
    list_of_df = []
    for index,ll in df.T.iterrows():
        res = cox_on_col_all(ll,df_surv[[duration_col,event_col]],cox_lfl,duration_col=duration_col,event_col=event_col)
    
        list_of_df.append(res)

    
    return pd.concat(list_of_df)


def cox_on_all_itterrows_parallel(df,df_surv,duration_col='pfs',event_col="pfs_event",num_processes=None):
    num_processes = num_processes or (cpu_count()-1)
    pool = Pool(num_processes)
    cox_lfl = CoxPHFitter()

    processes = [pool.apply_async(cox_on_col_all, args=(ll,df_surv[[duration_col,event_col]],cox_lfl,duration_col,event_col)) for index,ll in df.T.iterrows()]
    result = [p.get() for p in processes]

    
    return pd.concat(result)



def cox_on_all_itterrows_parallel_deepseek(df, df_surv, duration_col='pfs', event_col="pfs_event", num_processes=None):
    num_processes = num_processes or (cpu_count() - 1)
    pool = Pool(num_processes)

    # Create a list of arguments for each row
    args = [(ll, df_surv[[duration_col, event_col]], duration_col, event_col) for index, ll in df.T.iterrows()]

    # Use imap_unordered to apply the function to each row in parallel
    result = pool.imap_unordered(cox_on_col_all, args)

    # Collect the results into a list
    results_list = list(result)

    pool.close()
    pool.join()

    return pd.concat(results_list)


def plot_venn_preloaded(df1, df2, df3, df1_name, df2_name, df3_name, p_value_threshold, title, savepath=None):
    # Drop NaNs and find significant
    df1_nona = df1.dropna()
    df2_nona = df2.dropna()
    df3_nona = df3.dropna()

    print(df1_nona.shape)
    print(df2_nona.shape)
    print(df3_nona.shape)

    df1_signi = df1_nona[df1_nona['p'] < p_value_threshold]
    df2_signi = df2_nona[df2_nona['p'] < p_value_threshold]
    df3_signi = df3_nona[df3_nona['p'] < p_value_threshold]
    
    # Convert to sets for venn diagram
    df1_set = set(df1_signi.index)
    df2_set = set(df2_signi.index)
    df3_set = set(df3_signi.index)
    
    # Calculate intersections two at a time
    df1_df2_intersect = df1_set & df2_set 
    df1_df3_intersect = df1_set & df3_set 
    df2_df3_intersect = df2_set & df3_set 
    
    # Calculate intersection of three sets
    three_way_intersect = df1_set & df2_set & df3_set

    # Plot venn diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    fig.suptitle(title)
    venn_all = venn3(subsets=[set(df1.dropna().index), set(df2.dropna().index), set(df3.dropna().index)], 
                     set_labels=(df1_name, df2_name, df3_name), ax=ax1)
    ax1.title.set_text("MS analyzed")
    
    venn_signif = venn3(subsets=[df1_set, df2_set, df3_set], 
                        set_labels=(df1_name, df2_name, df3_name), ax=ax2)
    ax2.title.set_text(f"MS p.value < {p_value_threshold}")
    
    if savepath:
        directory = os.path.dirname(savepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(savepath)
    plt.show() 
  
    return df1, df2, df3, (df1_df2_intersect, df1_df3_intersect, df2_df3_intersect, three_way_intersect)




def plot_venn(df1, df2, df3, df1_name, df2_name, df3_name, p_value_threshold, title, savepath=None):

    
    # Load data from DataFrames
    df1 = pd.read_csv(df1,index_col=0)
    df2 = pd.read_csv(df2,index_col=0)
    df3 = pd.read_csv(df3,index_col=0)
    
    # Drop NaNs and find significant
    df1_nona = df1.dropna()
    df2_nona = df2.dropna()
    df3_nona = df3.dropna()

    print(df1_nona.shape)
    print(df2_nona.shape)
    print(df3_nona.shape)

    df1_signi = df1_nona[df1_nona['p'] < p_value_threshold]
    df2_signi = df2_nona[df2_nona['p'] < p_value_threshold]
    df3_signi = df3_nona[df3_nona['p'] < p_value_threshold]
    
    # Convert to sets for venn diagram
    df1_set = set(df1_signi.index)
    df2_set = set(df2_signi.index)
    df3_set = set(df3_signi.index)
    
    # Calculate intersections two at a time
    df1_df2_intersect = df1_set & df2_set 
    df1_df3_intersect = df1_set & df3_set 
    df2_df3_intersect = df2_set & df3_set 
    
    # Calculate intersection of three sets
    three_way_intersect = df1_set & df2_set & df3_set

    # Plot venn diagrams
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    
    fig.suptitle(title)
    venn_all = venn3(subsets=[set(df1.dropna().index), set(df2.dropna().index), set(df3.dropna().index)], 
                     set_labels=(df1_name, df2_name, df3_name), ax=ax1)
    ax1.title.set_text("MS analyzed")
    
    venn_signif = venn3(subsets=[df1_set, df2_set, df3_set], 
                        set_labels=(df1_name, df2_name, df3_name), ax=ax2)
    ax2.title.set_text(f"MS p.value < {p_value_threshold}")
    
    if savepath:
        directory = os.path.dirname(savepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(savepath)
    plt.show() 
  

  
    return df1,df2,df3, (df1_df2_intersect, df1_df3_intersect, df2_df3_intersect, three_way_intersect)



def plot_venn_HR_separated(df1, df2, df3, df1_name, df2_name, df3_name, p_value_threshold, title, savepath=None):
        # Load data from DataFrames
    df1 = pd.read_csv(df1,index_col=0)
    df2 = pd.read_csv(df2,index_col=0)
    df3 = pd.read_csv(df3,index_col=0)

    # for df1 df2 df3 with HR >1
    df1_deleter = df1[df1['exp(coef)']>1]
    df2_deleter = df2[df2['exp(coef)']>1]
    df3_deleter = df3[df3['exp(coef)']>1]

    # df 1 df2 df3 with HR < 1
    df1_benefique = df1[df1['exp(coef)']<1]
    df2_benefique = df2[df2['exp(coef)']<1]
    df3_benefique = df3[df3['exp(coef)']<1]
    
    if savepath:
        df1_deleter, df2_deleter, df3_deleter, intersect_deleter = plot_venn_preloaded(df1_deleter, df2_deleter, df3_deleter, df1_name, df2_name, df3_name, p_value_threshold, " HR > 1"+title, savepath+"_deletere.png")
        df1_benefique, df2_benefique, df3_benefique, intersect_benefique = plot_venn_preloaded(df1_benefique, df2_benefique, df3_benefique, df1_name, df2_name, df3_name, p_value_threshold, " HR < 1"+title, savepath+"_benefique.png")
    else:
        df1_deleter, df2_deleter, df3_deleter, intersect_deleter = plot_venn_preloaded(df1_deleter, df2_deleter, df3_deleter, df1_name, df2_name, df3_name, p_value_threshold, " HR > 1"+title, savepath=None)
        df1_benefique, df2_benefique, df3_benefique, intersect_benefique = plot_venn_preloaded(df1_benefique, df2_benefique, df3_benefique, df1_name, df2_name, df3_name, p_value_threshold, " HR < 1"+title, savepath=None)


    return intersect_benefique,intersect_deleter

# =============================================================================
# Création fonction qui somme les mutation pour chaque patient en inversant 0 et 1 si HR > 1
# =============================================================================

def calculate_norm_sum(df_table,df_cox):
    df = df_table.copy()
    df = df[df_cox.index]
    risk_index = (df_cox['exp(coef)']>1).index
    df[risk_index] = np.abs(1-df[risk_index])
    df = df.mean(axis=1)

    return df