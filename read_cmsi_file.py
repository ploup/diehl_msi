#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:22:13 2020

Script avec les differentes fonction permettantd de manipuler les fichier utilisé dans le projet CMSI

@author: aurelien
"""
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import scipy


from tqdm import tqdm

from matplotlib_venn import venn2

from matplotlib_venn import venn3

from venn import venn
from usefull_msi_script.diehl_msi.multi_tests import *
import numpy as np

import matplotlib.pylab as pylab
from scipy.stats import norm
from numpy.random import beta,binomial

from scipy.stats import kde

import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline, BSpline

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant




#%% boxplot


def mut_boxplot(df_mut_count, filt = 1, Titre = '',filtLength = None, anacol = 'norm'):
    """
    

    Parameters
    ----------
    df_mut_count : pd.DataFrame
        Dataframe généré a partir de mes script. Représente pour une cohorte donnée
        des infos pour chaque microsat séquencé. Une ligne par MS. 
        Colonnes indispensable : nb_presence : Nombre de patient séquencé
                                 length : taille du MS
                                 norm : fréquence de mutation (nb_mut/nb_presence)
    filt : float, entre O et 1. optional
        Quand filt diminue, le filtre est plus stringent. Par exemple
        # filt = 0.4 : on ne garde que les microsat observé dans plus de 40% des cas. The default is 1.
    Titre : str, optional
        Titre du boxplot. The default is ''.
    filtLength : TYPE, optional
        Filtre sur la taille du microsat. The default is None.
    anacol : TYPE, optional
        colonne a analyser. The default is 'norm'.

    Returns
    -------
    None.

    """

# anacol designe la colonne sur la quelle on fait les boxplot, par defaut nb de mut
    plt.close()

    df_mut_count = df_mut_count[df_mut_count['nb_presence']>= df_mut_count['nb_presence'].max()-(df_mut_count['nb_presence'].max()*filt)]
    
    if filtLength != None:
        df_mut_count = df_mut_count[df_mut_count['length']<= filtLength]
    
    
    if anacol == 'norm':
        boxplot = sns.boxplot(x=df_mut_count['length'],y=df_mut_count['norm'])
    else :
        df_freq = df_mut_count[anacol].div(df_mut_count['nb_presence'])
        boxplot = sns.boxplot(x=df_mut_count['length'],y=df_freq)
    plt.xlabel('Length')
    if filt !=0:
        plt.title(Titre+' with filt ')
    else:
        plt.title(Titre+' w/o filt ')
    plt.tight_layout()
    plt.ylim(0,1.1)
    plt.show()


def mut_boxplot_AllMS(All_MS,cohorte, filt = 1, Titre = '',filtLength = None):
# filt : entre O et 1. Quand filt diminue, le filtre est plus stringent. Par exemple
# filt = 0.4 : on ne garde que les microsat observé dans plus de 40% des cas
# anacol designe la colonne sur la quelle on fait les boxplot, par defaut nb de mut
    plt.close()
    
    if Titre == '':
        Titre = cohorte
    
    
    df_mut_count = All_MS[['length','mutfreq_'+cohorte,'nb_measure_'+cohorte,'mutfreq_'+cohorte]]
    df_mut_count.columns = ['length','mutfreq','nb_presence','norm']

    df_mut_count = df_mut_count[df_mut_count['nb_presence']>= df_mut_count['nb_presence'].max()-(df_mut_count['nb_presence'].max()*filt)]
    
    if filtLength != None:
        df_mut_count = df_mut_count[df_mut_count['length']<= filtLength]
    
    

    boxplot = sns.boxplot(x=df_mut_count['length'],y=df_mut_count['norm'])

    plt.xlabel('Length')
    plt.title(Titre)


    plt.tight_layout()
    plt.ylim(0,1.1)
    plt.show()


def lift_on_set(setchrid,fromdb,todb,SortirLeftover=False):
    """ Fonction pour traduire d'une version Hg à une autre en utilisant LiftOver.
    Sort le df traduit accompagné des lignes dupliqué (deux position dans une version peuvent etre placé à la même position dans l'autre version)
    /!\ Ne tournera pas sur un DataFrame déjà traduit
    return df,df_dup,leftover
    """

    
    from pyliftover import LiftOver
    lo = LiftOver(fromdb, todb)
    newset = set()
    leftover = []
    for chrid in setchrid:
        chrom,start = chrid.split('.')
        conv = lo.convert_coordinate(chrom,int(start))
        try:
            len(conv)
        except TypeError:
            leftover.append([ind,'Error'])
            continue
        if len(conv) >= 1:
            newset.add("{}.{}".format(conv[0][0],str(conv[0][1])))
        elif len(conv) ==0:
            leftover.append(chrid)


     

    if SortirLeftover:
        return newset,leftover
    else:
        return newset

def lift_on_df(df,chridcol='index',fromdb="hg38",todb="hg19"):
    """
    Fonction pour traduire d'une version Hg à une autre en utilisant LiftOver.
    
    Parameters
    ----------
    df : pd.DataFrame
        DF avec une colonnes ou un index avec des chrid d'un MS (chr.pos)
        /!\ Ne tournera pas sur un DataFrame déjà traduit.
    chridcol : str, optional
        colonnes avec chrid. The default is 'index'.
    fromdb : str, optional
        depuis quelle hg. The default is "hg38".
    todb : str, optional
        vers quel hg. The default is "hg19".

    Returns
    -------
    df : pd.DataFrame
        df traduit.
    df_dup : pd.DataFrame
        lignes dupliqué.
    leftover : pd.DataFrame
        (deux position dans une version peuvent etre placé à la même position dans l'autre version).

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
    
    from pyliftover import LiftOver
    lo = LiftOver(fromdb, todb)
    df = df.astype({'loc':'int64'})
    df[todbchridcol] = str()
    leftover = []
    
    for ind,row in tqdm(df.iterrows(),total=df.shape[0]):
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
    
    return df,df_dup,leftover


#%% Venn and intersection pas tres intéressant

def venn_sgd_outlier(df1,df2,df3,df4,Titre,labels):
    """
    OSEF: pour plotter un venn très spécifique: comparer la sortie du modèle CMGH
    deux deux run
    """
    #df1 and df2 : survivor 
    #df3 and df4 : transformator
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,7))
    fig.suptitle(Titre)


    v2 = venn2(subsets=[set(df1.index), set(df2.index)],set_labels=[labels[0]+' survivor',labels[1]+' survivor'],ax=ax1)


    v2.get_patch_by_id('10').set_color('yellow')
    v2.get_patch_by_id('01').set_color('red')
    v2.get_patch_by_id('11').set_color('orange')

    v2.get_patch_by_id('10').set_edgecolor('none')
    v2.get_patch_by_id('01').set_edgecolor('none')
    v2.get_patch_by_id('11').set_edgecolor('none')

    v2.get_label_by_id("11").set_y(0.05)

#     set1size = int(v2.get_label_by_id('10').get_text())
#     setsharedsize = int(v2.get_label_by_id('11').get_text())
#     set2size = int(v2.get_label_by_id('01').get_text())    


    v2 = venn2(subsets=[set(df3.index), set(df4.index)],set_labels=[labels[0]+' transformator',labels[1]+' transformator'],ax=ax2)


    v2.get_patch_by_id('10').set_color('yellow')
    v2.get_patch_by_id('01').set_color('red')
    v2.get_patch_by_id('11').set_color('orange')

    v2.get_patch_by_id('10').set_edgecolor('none')
    v2.get_patch_by_id('01').set_edgecolor('none')
    v2.get_patch_by_id('11').set_edgecolor('none')

    v2.get_label_by_id("11").set_y(0.05)

def venn3_diag_chrid(df1,df2,df3,labels=('A','B','C'),plot=True):
    
    
    
    a =set(df1.index)
    b= set(df2.index)
    c= set(df3.index)
    
    if plot:
        plt.figure(figsize=(5,5))
        v3 = venn3(subsets=[a, b,c],set_labels=labels)
        plt.title('Venn diagram of '+labels[0]+', '+labels[1]+' and '+labels[2])
    
    abc = a.intersection(b,c)
    ab = a.intersection(b)
    ac = a.intersection(c)
    bc = b.intersection(c)
    union = a.union(b,c)
    
    
    return abc,ab,ac,bc,union


def venn3_venpack(df1,df2,df3,labels=['A','B','C']):
    
    
    
    a =set(df1.index)
    b= set(df2.index)
    c= set(df3.index)
    
    dico = {labels[0]:a,labels[1]:b,labels[2]:c}
    
    venn(dico)
    
    abc = a.intersection(b,c)

    
    
    return abc


def intersect_outlier(df1,df2,df3,labels=('A','B','C')):
    
    abc,ab,ac,bc,union = venn3_diag_chrid(df1,df2,df3,labels=labels,plot=False)
    
    df_cross = pd.DataFrame(index=union,columns=labels)
    
    
    for df,label in zip([df1,df2,df3],labels):
        df_cross.loc[df_cross[df_cross.index.isin(df.index)].index,label]=1
        df_cross.loc[df_cross[~df_cross.index.isin(df.index)].index,label]=0

    
    df_cross['score'] = df_cross.sum(axis=1).astype(int)
    
    return df_cross



def pair_col_compare(df1,df2,df1_cols=[0,1],df2_cols=[0,1]):
    df1['temp'] = df1[df1_cols[0]].astype(str)+df1[df1_cols[1]].astype(str)
    df2['temp'] = df2[df2_cols[0]].astype(str)+df2[df2_cols[1]].astype(str)
    diff = df1[~df1['temp'].isin(df2['temp'])].drop('temp',1)
    sim = df1[df1['temp'].isin(df2['temp'])].drop('temp',1)
    return diff,sim

def venn_diag_chrid(df1,df2,df1_cols=[0,1],df2_cols=[0,1]):
    df1['temp'] = df1[df1_cols[0]].astype(str)+df1[df1_cols[1]].astype(str)
    df2['temp'] = df2[df2_cols[0]].astype(str)+df2[df2_cols[1]].astype(str) 
    
    venn2([set(df1['temp']), set(df2['temp'])])
#%% filt and stat

def filt_numcount(df_mut_count,filtLength=25,filt=10):
    df_mut_count = df_mut_count[df_mut_count['length']<= filtLength]
# pour filtrer en proportion de presence    df_mut_count = df_mut_count[df_mut_count['nb_presence']>= df_mut_count['nb_presence'].max()-(df_mut_count['nb_presence'].max()*filt)]
    df_mut_count = df_mut_count[df_mut_count['nb_presence']>=filt] #filtre avec un nb mini de presence (defaut 10)
    
    return df_mut_count



def outlier_identification(df_mut_count,filt = 0,filtLength = None):
    """
    Dans la cadre le l'etude de la freq de mutation en fonction de la taille:

    définie les outliers sur la base des quartiles supérieur et inférieur
    ----------
    df_mut_count : TYPE
        DESCRIPTION.
    filt : TYPE, optional
        DESCRIPTION. The default is 0.
    filtLength : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df_outlier : TYPE
        DESCRIPTION.

    """
    df_mut_count = df_mut_count[df_mut_count['nb_presence']>= df_mut_count['nb_presence'].max()-(df_mut_count['nb_presence'].max()*filt)]
    
    if filtLength != None:
        df_mut_count = df_mut_count[df_mut_count['length']<= filtLength]
    
    
    length_q1 = df_mut_count.groupby(df_mut_count['length']).quantile(0.25)['norm']
    length_q3 = df_mut_count.groupby(df_mut_count['length']).quantile(0.75)['norm']
    outlier_top_lim = length_q3 + 1.5 * (length_q3 - length_q1)
    outlier_bottom_lim = length_q1 - 1.5 * (length_q3 - length_q1)
    
    outlier_list = []
    
    # genere la liste des outliers pour les comparer avec CMGH
    df_outlier = pd.DataFrame(columns=['chrid','quantile','norm','length','Type'])
    for row in tqdm(df_mut_count.itertuples(), total=df_mut_count.shape[0]):
        length = row.length 
        val = row.norm
        chrid = row.Index
        df_len = len(df_outlier)
        if val < outlier_bottom_lim.loc[length]:
            outlier_list.append([chrid,outlier_bottom_lim.loc[length],val,length,'Negative selection'])
            df_outlier.loc[df_len] = [chrid,outlier_bottom_lim.loc[length],val,length,'Negative selection']
        if val > outlier_top_lim.loc[length]:
            outlier_list.append([chrid,outlier_top_lim.loc[length],val,length,'Positive selection'])
            df_outlier.loc[df_len] = [chrid,outlier_bottom_lim.loc[length],val,length,'Negative selection']


    outlier_loc_list = [item[0].split('.') for item in outlier_list]
    
    df_out = pd.DataFrame(outlier_loc_list,columns=['chromosome','location'])
    df_out = df_out.astype({'chromosome':str,'location':int})
    
    df_outlier[['chromosome','location']] = df_out
    
    df_outlier['end'] = df_outlier['length']+df_outlier['location']
    df_outlier = df_outlier.astype({'end':int})


    return df_outlier

def pearsoncorrelation_cohortmutfreq(df_mut1_freq,df_mut2_freq):
    df1_in_df2 = df_mut1_freq[df_mut1_freq.index.isin(df_mut2_freq.index)].sort_index()
    df2_in_df1 = df_mut2_freq[df_mut2_freq.index.isin(df_mut1_freq.index)].sort_index()
    
    return scipy.stats.pearsonr(df1_in_df2, df2_in_df1)


def correlation_cohortmutfreq(df_mut1_freq,df_mut2_freq,sbplt=None,xlabel = 'cohort1',ylabel='cohort2',ci=95,Titre = ''):
    """ Sort un plot d'une valeur (à la bas la freq) plotter contre son equivalant dans une autre cohorte
    en entrée ce sont des Series avec pour index le chrid.
    
    """
    df1_in_df2 = df_mut1_freq[df_mut1_freq.index.isin(df_mut2_freq.index)].sort_index()
    df2_in_df1 = df_mut2_freq[df_mut2_freq.index.isin(df_mut1_freq.index)].sort_index()
    
    if sbplt is not None:
        reg = sns.regplot(x=df1_in_df2, y=df2_in_df1,ci=ci,ax = sbplt,
        line_kws={"color":"r","alpha":0.7,"lw":5})
    else:
        reg = sns.regplot(x=df1_in_df2,
                          y=df2_in_df1
                          ,ci=ci,
                          line_kws={"color":"r","alpha":0.7,"lw":5})
    
    reg.set_title(Titre)
    
    reg.set_xlabel(xlabel)
    reg.set_ylabel(ylabel)
    reg.set_ylim(0,1)
    reg.set_xlim(0,1)
    
    return scipy.stats.pearsonr(df1_in_df2, df2_in_df1)


#%%

def feq_vs_freq(df1,df2,freqcol1,freqcol2):
    """
    plot la difference de frequence de mutation entre deux cohortes

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    freqcol1 : TYPE
        DESCRIPTION.
    freqcol2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df1_in_df2 = df1[df1.index.isin(df2.index)]
    df2_in_df1 = df2[df2.index.isin(df1.index)]
    
    df2_in_df1  = df2_in_df1.sort_index()
    df1_in_df2 = df1_in_df2.sort_index()
    
    sns.regplot(x=df1_in_df2[freqcol1], y=df2_in_df1[freqcol2],ci=99)
    
    plt.ylim(0,1)
    plt.xlim(0,1)    
    
    return

   
    
    
def filtnan(tabmut,filtcoef=0.5): 
    """
    drop les colonnes ayant plus que (coeff * nb de ligne) de nan (1 très strigent, 0 pas du tout)

    Parameters
    ----------
    tabmut : pd.DataFrame
        chaque ligne un patient, chaque colonne une postion. 0 pour wt et 1 pour muté.
    filtcoef : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    #drop les colonnes ayant plus que (coeff * nb de ligne) de nan (1 très strigent, 0 pas du tout)
    filt_nan = len(tabmut)*filtcoef
    return tabmut.loc[:,(tabmut.isna().sum() <= filt_nan)]     
    
def extract_colname(df,string_list):


    string_cols = ['Func.refGene', 'Gene.refGene', 'length', 'nucleo']
    string_cols += [col for string in string_list  for col in df.columns if string in col]

    return df[string_cols]
    

    
def extract_colname_mut_mat(df,string_list):
    
    string_cols = []
    string_cols += [col for string in string_list  for col in df.columns if string in col]

    return df[string_cols]    
#%% Annotation foncionnelle


# def annot(entry,annovpath):
#     ## ???
#     annovfile = pd.read_csv(annovpath)
#     annovfile.index = annovfile["Chr"].astype(str)+'.'+annovfile['Start'].astype(str)
    
#     if type(entry) is pd.DataFrame:
#         return entry.join(annovfile[['Func.refGene','Gene.refGene']],how='left')
    
#     elif type(entry) is set:
#         return annovfile.loc[entry]
    
#     else:
#         print('Did not recognize the input type')
#         print(type(entry))
        

# def annot_df_mut(df_mut,annovpath):

    
#     return df_mut.join(annovfile.set_index('Start')[['Func.refGene','Gene.refGene']],on='location',how='left')
    
# def annot_oulier(outl,annovpath):
#     annovfile = pd.read_table(annovpath,sep='\t')
#     annovfile['chrid'] = annovfile['Chr'].astype(str)+'.'+annovfile['Start'].astype(str)
#     return outl.join(annovfile.set_index('chrid')[['Func.refGene','Gene.refGene']],on='chrid',how='left')



# def annot_cox(df_mut,annovpath):
#     annovfile = pd.read_table(annovpath,sep='\t')
#     annovfile['chrid']= annovfile["Chr"].astype(str)+'.'+annovfile['Start'].astype(str)
#     return df_mut.join(annovfile.set_index('chrid')[['Func.refGene','Gene.refGene']],on='location',how='left')

def annot_intersect(intersect,annovpath):
    annovfile = pd.read_table(annovpath,sep='\t')
    annovfile.index = annovfile["Chr"].astype(str)+'.'+annovfile['Start'].astype(str)
    return annovfile.reindex(list(intersect))


def annot_ms(list_MS):
    import mysql.connector as mariadb

    if len(list_MS) == 0:
        print("Empty MS list")
        return

    locus_list = tuple(element.replace(".",":") for element in list_MS)
    # Connect to the DB
    connection = mariadb.connect(
        user="consulter",
        password="insermU938",
        database="msi",
        host="134.157.67.8",
        port=3310
    )
    cursor = connection.cursor()

    if len(locus_list) == 1:
        cursor.execute(f"SELECT * FROM ms_bank WHERE locus = '{locus_list[0]}'")
    else:
        # Execute the query
        cursor.execute(f"SELECT * FROM ms_bank WHERE locus in {locus_list}")


    # # Close the connection
    

    ms_df = pd.DataFrame(cursor.fetchall(),columns=cursor.column_names)
    
    connection.close()
    return ms_df

#%% Fonction model CMGH


def outlierCMGH(df_mut_count,coefpath,Titre,k=1,saveif=False,savepath='',boxplot=False,intersectionif=False,df_cross=''):
    if saveif:
        if savepath == '':
            print('Please specify a path to save the outliers')
            return
    coef_hat = np.load(coefpath+Titre+'coef_hat'+str(1)+'.npy')
    
    df_mut_count['chrid']=df_mut_count.index
    nmeasured = df_mut_count.nb_presence.values # nb_presence
    # fonction = data_measured['Func.refGene']# n_j fonction du gène annovar
    prob = df_mut_count.norm.values         # freq de mut
    rpt = df_mut_count.length.values    #   length du microsat
    refgene = df_mut_count.chrid.values     #    chrid
    # genename = data_measured.genename.values   # gene name
    m = df_mut_count.shape[0]

    
    alpha_hat = coef_hat[1]
    beta1_hat = coef_hat[2]
    beta2_hat = coef_hat[3]
    c_hat = coef_hat[0]
    exp_hat = np.exp(beta1_hat+beta2_hat*rpt)
    P_hat = np.exp(alpha_hat)*exp_hat/(1+exp_hat)#+intensity)/(1+np.exp(beta1star+beta2star*rpt_array+intensity))
    
    phi_hat = 1/(c_hat+1)
    
    gamma = P_hat*c_hat
    delta = (1-P_hat)*c_hat
    
    
    norm_residuals = np.sqrt(nmeasured)*(prob - P_hat)/np.sqrt((1+phi_hat)*P_hat*(1-P_hat))
    
    p_values = list(2-2*norm.cdf(np.abs(norm_residuals)))
    
    alpha = 0.10
    crit_BH = BH(np.sort(p_values),alpha)
    crit_BY = BY(np.sort(p_values),alpha)
    
    dat_intensity = df_mut_count.loc[:,["chrid",'norm','length','nb_presence']]
    
    dat_intensity['pval']  = pd.Series(p_values,name='pval',index=df_mut_count.index)
    

    
    dat_intensity['resid']  = pd.Series(norm_residuals,name='resid',index=df_mut_count.index)
    
    data_crit = dat_intensity.loc[dat_intensity['pval']<=crit_BY]
    transformators= data_crit.loc[data_crit["resid"] >0]
    survivors= data_crit.loc[data_crit["resid"] <0]
    print('Transfo shape : ',transformators.shape)
    print('Survivor shape :',survivors.shape)
    
    if saveif:
        transformators.sort_values('pval',axis=0,ascending=True).to_excel(savepath+Titre+str(k)+'transformators'+str(alpha)+'.xls')
        survivors.sort_values('pval',axis=0,ascending=True).to_excel(savepath+Titre+str(k)+'survivors'+str(alpha)+'.xls')
        
    if boxplot:
   
        
        
        if intersectionif:
            
            if type(df_cross) == str:
                print('Please enter a cross dataframe.')
                return
            
            
            pylab.rcParams['figure.figsize'] = 24, 22  # image size
            max_repeat =  rpt.max()
            datanR = []
            # grid = np.linspace(5, max_repeat, 30)
            plt.xlim([4.90,29])
            plt.ylim([-0.01,1.02])
            plt.subplot(211)
            #plt.plot(grid,np.exp(beta[0]+beta[1]*grid)/(1+np.exp(beta[0]+ beta[1]*grid)),'b-')
            
            #plt.plot(grid,np.exp(alpha_hat+beta1_hat+beta2_hat*grid)/(1+np.exp(beta1_hat+beta2_hat*grid)),'r-')
            
            
            for nrepeat in range(5,max_repeat):
                datanR.append( prob[np.where(rpt.reshape(rpt.shape[0],) == nrepeat)])
            print('length data simulée : ',len(datanR))
            
            smoothed(df_mut_count, coefpath, Titre)
            #plt.boxplot(datanR,positions=list(np.arange(5,max_repeat,1)),showfliers=False)
            

            
            
            df_cross = pd.merge(df_cross,df_mut_count[['length','norm']],right_index=True,left_index=True)
            
            intersect = df_cross[df_cross['score']==3]
            
            pt1 = plt.scatter(intersect['length'],intersect['norm'],marker='^',edgecolors='k',c="None",label='Outlier in the 3 cohorts')
            
            i=0
            for col in df_cross.columns[0:3]:
                if col != Titre :
                    i+=1
                    tmp = df_cross[(df_cross[Titre]==1)&(df_cross[col]==1)&(df_cross['score']==2)]
                    if i == 1:
                        pt2 = plt.scatter(tmp['length'],tmp['norm'],marker='*',edgecolors='crimson',c="None",label='Shared with '+col)
                    if i == 2:
                        pt3 = plt.scatter(tmp['length'],tmp['norm'],marker='*',edgecolors='m',c="None",label='Shared with '+col)
                else :
                    tmp = df_cross[(df_cross[Titre]==1)&(df_cross['score']==1)]
                    pt4 = plt.scatter(tmp['length'],tmp['norm'],marker='+',c='limegreen',label='Only in '+Titre)
                
            plt.legend(handles=[pt1,pt2,pt3,pt4], loc='lower right')

        
        else:
            
            
            pylab.rcParams['figure.figsize'] = 24, 22  # image size
            max_repeat =  rpt.max()
            datanR = []
            # grid = np.linspace(5, max_repeat, 30)
            plt.xlim([4.90,29])
            plt.ylim([-0.01,1.02])
            plt.subplot(211)
            #plt.plot(grid,np.exp(beta[0]+beta[1]*grid)/(1+np.exp(beta[0]+ beta[1]*grid)),'b-')
            
            #plt.plot(grid,np.exp(alpha_hat+beta1_hat+beta2_hat*grid)/(1+np.exp(beta1_hat+beta2_hat*grid)),'r-')
            
            
            for nrepeat in range(5,max_repeat):
                datanR.append( prob[np.where(rpt.reshape(rpt.shape[0],) == nrepeat)])
            print('length data simulée : ',len(datanR))
            
            smoothed(df_mut_count, coefpath, Titre)
            #plt.boxplot(datanR,positions=list(np.arange(5,max_repeat,1)),showfliers=False)
            

        
                
            

            pt1 = plt.scatter(transformators['length'].values, transformators['norm'].values, c='r',label="Transformators")#, s=transformators_ada['intensity'].values,  alpha=0.5)
            pt2 = plt.scatter(survivors['length'].values, survivors['norm'].values, c="b",label="Survivors")#, s=survivors_ada['intensity'].values,  alpha=0.5)
            
            plt.legend(handles=[pt1,pt2], loc='lower right')
        
        plt.xlabel('Repeat times')
        plt.ylabel('%mutated')
       
        plt.title(Titre)
        
    return survivors,transformators




def smoothed(df_mut_count,coefpath,Titre,nbins=200,n_sim=2000):
    
    
    coef_hat = np.load(coefpath+Titre+'coef_hat'+str(1)+'.npy')
    df_mut_count['chrid']=df_mut_count.index
    nmeasured = df_mut_count.nb_presence.values # nb_presence
    # fonction = data_measured['Func.refGene']# n_j fonction du gène annovar
    prob = df_mut_count.norm.values         # freq de mut
    rpt = df_mut_count.length.values    #   length du microsat
    refgene = df_mut_count.chrid.values     #    chrid
    # genename = data_measured.genename.values   # gene name
    m = df_mut_count.shape[0]
    max_repeat =  rpt.max()
    print(m)
    print(np.max(rpt))
    
    alpha_hat = coef_hat[1]
    beta1_hat = coef_hat[2]
    beta2_hat = coef_hat[3]
    c_hat = coef_hat[0]
    
    
    #n_sim = rpt.shape[0]
    x= np.random.uniform(5,max_repeat,n_sim)
    P = np.exp(alpha_hat+beta1_hat+beta2_hat*x)/(1+np.exp(beta1_hat+beta2_hat*x))#np.exp(alpha_hat+beta1_hat+beta2_hat*rpt)/(1+np.exp(beta1_hat+beta2_hat*rpt))#+intensity)/(1+np.exp(beta1star+beta2star*rpt_array+intensity))
    
    
    gamma = P*c_hat
    delta = (1-P)*c_hat

    #n_sim = rpt.shape[0]
    Y = np.zeros(n_sim)
    for k in np.arange(n_sim):
        P_rand = beta(gamma[k],delta[k])
        Y[k] = binomial(nmeasured[k],P_rand)
    prob_sim = Y/nmeasured[0:n_sim]
    #rpt = rpt[0:n_sim]
    
    
    datanR = []
    datanR_sim = []

    
    datanR_sim = pd.DataFrame(np.array([x,prob_sim]).T,columns=['length','norm'])
    
    datanR_sim['length'] = pd.cut(x=datanR_sim['length'],bins=np.arange(4.5,25.5,1),labels=np.arange(5,25,1))
    
    datanR_sim.length = datanR_sim.length.astype(np.int64)
    
    for nrepeat in range(5,max_repeat):
        datanR.append( prob[np.where(rpt.reshape(rpt.shape[0],) == nrepeat)])
    
    # x=[]
    # y=[]
    # for i in range(len(datanR_sim)):
    #     for t in range(len(datanR_sim[i])):
    #         y.append(datanR_sim[i][t])
    #         x.append(i+5)
        
    #nbins = 200
    



    grid = np.linspace(5, max_repeat, 30)
    plt.xlim([4.90,max(rpt)])
    plt.ylim([-0.01,1.02])
    k = kde.gaussian_kde([x,prob_sim])
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, prob_sim.min():prob_sim.max():nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape),shading='gouraud',cmap=plt.cm.afmhot_r)
    plt.colorbar()
    #plt.contour(xi, yi, zi.reshape(xi.shape) )
    #plt.plot(grid,np.exp(alpha_hat+beta1_hat+beta2_hat*grid)/(1+np.exp(beta1_hat+beta2_hat*grid)),'r-')
    
    length_q1 = datanR_sim.groupby(datanR_sim['length']).quantile(0.1)['norm']
    length_q3 = datanR_sim.groupby(datanR_sim['length']).quantile(0.9)['norm']
    median = datanR_sim.groupby('length').median()['norm']
    
    
   # traçage de la courbe quantile 0.1  simulé
    
    xq1 = np.linspace(length_q1.index.min(),length_q1.index.max(),300)
    spl_q1 = make_interp_spline(length_q1.index, length_q1, k=3)
    q1_smooth = spl_q1(xq1)
 
    q1 = plt.plot(xq1,q1_smooth,c='k',linestyle='dashed', label='Quantile [0.1,0.9]')
    
# traçage de la courbe quantile 0.9  simulé

    xq3 = np.linspace(length_q3.index.min(),length_q3.index.max(),300)
    spl_q3 = make_interp_spline(length_q3.index, length_q3, k=3)
    q3_smooth = spl_q3(xq3)

    plt.plot(xq3,q3_smooth,c='k',linestyle='dashed')
    
    
    
    # Médiane simulée
    
    
    xmed = np.linspace(median.index.min(),median.index.max(),300)
    spl_med = make_interp_spline(median.index,median,k=3)
    splmed_smooth = spl_med(xmed)
        
    med1 = plt.plot(xmed,splmed_smooth,c='b',label='Model')
    plt.scatter(median.index,median,c='b')
    
    
    #Médiane observée
    
    
    median_obs = df_mut_count.groupby('length').median()['norm']
    
    xmed_obs = np.linspace(median_obs.index.min(),median_obs.index.max(),300)
    spl_med_obs = make_interp_spline(median_obs.index,median_obs,k=3)
    splmed_obs_smooth = spl_med_obs(xmed_obs)
    
    med2 = plt.plot(xmed_obs,splmed_obs_smooth,c='limegreen',label = 'Observed')
    plt.scatter(median_obs.index,median_obs,c='limegreen')

    
   
    #plt.plot(x,prob_sim,'ko')
    #plt.hist2d(x, prob_sim, bins=nbins)



    #plt.boxplot(datanR_sim,positions=list(np.arange(5,max_repeat,1)))

    first_legend = plt.legend(loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.title("Données Simulées")
    
    #plt.show()

