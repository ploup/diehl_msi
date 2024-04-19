#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:40:58 2022
All function and classes for GridSearchSimpleImputerKbestCoxRSF.py


Création de class héritante d'autre class dans le but de sortir des Dataframe
grandement inspiré de https://github.com/benman1/OpenML-Speed-Dating/blob/master/openml_speed_dating_pipeline_steps/openml_speed_dating_pipeline_steps.py
Court sur les class https://www.programiz.com/python-programming/inheritance

@author: aurelien
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.compare import compare_survival
import warnings
from scipy.stats import loguniform
# from bio_corex import corex as ce



#%%
# invert 0 and 1 if the hasard ratio shows a benefit of the MS mutation
def invert01(ll):
    if ll == 0.:
        return 1
    elif ll == 1.:
        return 0
    else:
        return ll
    
def invertcol01(col):
    return col.apply(invert01)
#%%
def groups_gen_TIMSIGA(X_index):
    """
    from the pd.DatFrame.index of the TIMSIGA mutation profile generates groups
    array depending on samp name (here X index). This array is use for the
    group GroupKFold sklearn function

    Parameters
    ----------
    X_index : pd.Dataframe.index
        index of the Mutation matrix of TIMSIGA.

    Returns
    -------
    TYPE np.array
        array with 1 for IMSI, 2 for TMSI and 3 for TCGA.

    """
    groups = []
    for i in X_index :
        if 'TCGA' in i:
            groups.append(3)
        elif "IMSI" in i:
            groups.append(1)
        elif "CORE" in i or "TMSI" in i:
            groups.append(2)
        else:
            groups.append(3)
    
    return np.array(groups)
    #%% ftsel class create
    

"""
Création de Class de feature selection simple intégrable à un pipeline sklearn
"""

# The CoxAndMutfreqFtSel class inherits from the sklearn.base classes 
# (BaseEstimator, TransformerMixin). This makes it compatible with 
# scikit-learn’s Pipelines


class MutfreqFtSel(BaseEstimator, TransformerMixin):
    """
    Select feature with a minimum and maximum of mutation
    
    Parameters
    ---
    mini_counts = threshold value of selection. keep MS with more than mini_count mutation
    and with less than cohorte size - mini_count mutation
    """
    # initializer 
    def __init__(self, mini_count=1):
        self.mini_count = mini_count
        return None
        # save the features list internally in the class
        
    def fit(self, X, y=None):
        self.mutfreq = X.mean()
        self.counts = X.count()
        self.max = X.max()
        self.feature_names_in_ = X.columns
        return self
    def transform(self, X, y = None):
        check_is_fitted(self,['mutfreq','counts','max'])
        # return the dataframe with the specified features
        X2 = X.loc[:,(self.mutfreq >= self.mini_count/self.counts)&(self.mutfreq <= (self.counts-self.mini_count)/self.counts)]
        self.feature_names_out_ = X2.columns
        return X2
    
    def get_feature_names_out(self):
        return self.feature_names_out_
    
    
class MutfreqFtSelDelsize(BaseEstimator, TransformerMixin):
    """
    Select feature with a minimum and maximum of mutated position
    
    Parameters
    ---
    mini_counts = threshold value of selection. keep MS with more than mini_count mutation
    and with less than cohorte size - mini_count mutation
    """
    # initializer 
    def __init__(self, mini_count=1):
        self.mini_count = mini_count
        return None
        # save the features list internally in the class
        
    def fit(self, X, y=None):
        
        self.nonzero_count = X[X!=0].count()
        self.counts = X.count()
        self.mutfreq = self.nonzero_count/self.counts
        self.max = X.max()
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X, y = None):
        check_is_fitted(self,['mutfreq','counts','max'])
        # return the dataframe with the specified features
        X2 = X.loc[:,(self.mutfreq >= self.mini_count/self.counts)&(self.mutfreq >= 1-self.mini_count/self.counts)]
        self.feature_names_out_ = X2.columns
        return X2
    
    def get_feature_names_out(self):
        return self.feature_names_out_


class FilterNA(BaseEstimator,TransformerMixin):
    """
    Filter out position with too much NA
    
    Parameters
    ---
    NAmax : maximum rate of NA allow over all position
    """
    def __init__(self,NAmax=0.05):
        self.NAmax = NAmax
        
    def fit(self, X, y=None):
        # calculate the missing data (NA) rate per columns
        self.narate = X.isna().sum()/X.shape[0]
        self.feature_names_in_ = X.columns
        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self, ['narate'])
        
        X = X.loc[:,self.narate<=self.NAmax] # return Dataframe with NA rate lower than NAmax
        
        self.feature_names_out_ = X.columns
        return X
    
    def get_feature_names_out(self):
        return self.feature_names_out_


#%% Class adaptation

    
class SimpleImputerMostFreq(SimpleImputer):
    """
    Comme SimpleImputer de sklearn mais conserve à la sortie des noms de 
    feature et sort un Dataframe. Peut être obsolète avec la dernière version 
    de sklearn.
    Voir doc sklearn
    """
    def __init__(self,strategy="most_frequent"):
                            
        SimpleImputer.__init__(self,strategy=strategy)
    
    def fit(self, X, y=None):
        super().fit(X, y)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.features = list(X.columns)
        else:
            self.features = list(range(X.shape[1]))
        return self
        
    def transform(self,X):
        X2 = super().transform(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(
                data=X2, 
                columns=self.get_feature_names(),
                index=X.index
            
            )
        else:
            return X2
                            
    def get_feature_names(self):
        return self.features


class SelectKBestWithName(SelectKBest):
    """
    class pour adapter la class sklearn SelectKBest pour garder les features
    names et le format Dataframe. Peut être obsolète avec la dernière version 
    de sklearn.
    Voir doc sklearn
    """

    def __init__(self,score_func,k=10):
        SelectKBest.__init__(self,score_func=score_func,k=k)
        
    def fit(self,X,y):

   

        super().fit(X,y)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            self.features = list(X.columns)
        else:
            self.features = list(range(X.shape[1]))
            
        return self
    
    def transform(self, X):
        """Select Kbest  in X. Returns a DataFrame if given
        a DataFrame.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """
        
        X2 = super().transform(X)
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return pd.DataFrame(
                data=X2, 
                columns=super().get_feature_names_out(),
                index=X.index
            )
        else:
            return X2




class SelectKBestCox( BaseEstimator, TransformerMixin):
    """
    Class sklearn compatible permettant de conserver les k meilleur position muté
    sur la base d'une regression de cox univariée. --> on garde les K meilleures c-index
    """
    
    def __init__(self, alpha=0.1, k=200, random=False, pvalue = False):
        self.alpha = alpha
        self.k = k
        self.pvalue = pvalue
        # permet de tester si la restriction au hasard de MS est aussi efficace
        self.random = random
        

        
    def fit(self,X,y):
        """
        https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
        calculate the Cox score for each feature of a given X,y sksurv data
        
        revisite du selectKbest with featurename

        Parameters
        ----------
        X : (array-like, shape = (n_samples, n_features)) 
            features matrix, rows <=> patients.
        y : (structured array, shape = (n_samples,))
            A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field..

        Returns
        -------
        scores : (np.arrays, shape = (n_features))
            for each feature return Cox score.

        """
        if self.random:
            self.score_ = None
            return self
        
            
        else:
            X_col = X.columns
            X = X.values
            n_features = X.shape[1]
            scores = np.empty(n_features)
            error = list()
            succes = list()
            m = CoxPHSurvivalAnalysis(alpha=self.alpha)
            #suppress warnings
            # warnings.filterwarnings('ignore')
            for j in range(n_features):
                Xj = X[:, j:j+1]
                nabool = ~np.isnan(Xj.astype(float))
                
                
                if self.pvalue:
                    try:
                        scores[j] = compare_survival(y[np.argwhere(nabool)[:,0]],Xj[nabool])[1]
                        succes.append(X_col[j])
                    except :
                        scores[j] = np.nan
                        error.append(X_col[j])
                    
                else :
                    try :
                        m.fit(Xj[nabool].reshape(-1,1), y[np.argwhere(nabool)[:,0]])
                        scores[j] = m.score(Xj[nabool].reshape(-1,1), y[np.argwhere(nabool)[:,0]])
                        succes.append(X_col[j])
                    except:
                        error.append(X_col[j])
                        scores[j] = np.nan
                        continue
                    

            # warnings.filterwarnings('default')

            self.score_ = pd.Series(scores,index=X_col)
            self.cox_fit_fail_ = error
            self.cox_fit_success_ = succes
            
            return self
            
    
    def transform(self, X):
        """Select Kbest  in X. Returns a DataFrame if given
        a DataFrame.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """

        
        if self.random :
            X = X.sample(n=self.k, axis = "columns")
            self.feature_names_out_ = X.columns
            return X
        elif self.pvalue :
            K_best_col = self.score_.sort_values(ascending=True)[:self.k].index
            self.feature_names_out_ = K_best_col
            return X[K_best_col]
        
        else : 
            K_best_col = self.score_.sort_values(ascending=False)[:self.k].index
            self.feature_names_out_ = K_best_col
            return X[X.columns.intersection(set(K_best_col))]
        
    def get_feature_names_out(self):
        return self.feature_names_out_

      
class SelectKBestCox_invert_mut( BaseEstimator, TransformerMixin):
    """
    Comme SelectKBestCox mais inverse note 1 pour wt et 0 pour les muté sur les 
    postions où l'hazard ratio montre un effet bénéfique. Permet en fin de coourse
    de tester une moyenne
    """
    def __init__(self, alpha=0, k=200, random=False):
        self.alpha = alpha
        self.k = k
        # permet de tester si la restriction au hasard de MS est aussi efficace
        self.random = random
        

        
    def fit(self,X,y):
        """
        https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
        calculate the Cox score for each feature of a given X,y sksurv data
        
        revisite du selectKbest with featurename

        Parameters
        ----------
        X : (array-like, shape = (n_samples, n_features)) 
            features matrix, rows <=> patients.
        y : (structured array, shape = (n_samples,))
            A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field..

        Returns
        -------
        scores : (np.arrays, shape = (n_features))
            for each feature return Cox score.

        """
        if self.random:
            self.score_ = None
            return self
        
            
        else:
            X_col = X.columns
            X = X.values
            n_features = X.shape[1]
            scores = np.empty(n_features)
            hasard = np.empty(n_features)
            error = list()
            succes = list()
            m = CoxPHSurvivalAnalysis(alpha=self.alpha)
            #suppress warnings
            # warnings.filterwarnings('ignore')
            for j in range(n_features):
                Xj = X[:, j:j+1]
                nabool = ~np.isnan(Xj.astype(float))

                    

                try :
                    m.fit(Xj[nabool].reshape(-1,1), y[np.argwhere(nabool)[:,0]])
                    scores[j] = m.score(Xj[nabool].reshape(-1,1), y[np.argwhere(nabool)[:,0]])
                    hasard[j] = m.coef_[0]
                    succes.append(X_col[j])
                except:
                    error.append(X_col[j])
                    scores[j] = np.nan
                    hasard[j] = np.nan
                    continue
                    

            # warnings.filterwarnings('default')

            self.score_ = pd.Series(scores,index=X_col)
            self.coef_ = pd.Series(hasard,index=X_col)
            self.cox_fit_fail_ = error
            self.cox_fit_success_ = succes
            
            return self
            
    
    def transform(self, X):
        """Select Kbest  in X. Returns a DataFrame if given
        a DataFrame.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input data to complete.
        """

        
        if self.random :
            X = X.sample(n=self.k, axis = "columns")
            self.feature_names_out_ = X.columns
            return X

        else : 
            K_best_col = self.score_.sort_values(ascending=False)[:self.k].index
            self.feature_names_out_ = K_best_col
            self.inverted_col_ = self.coef_[self.coef_<0].index
            return X[K_best_col].apply(lambda x: invertcol01(x) if x.name in self.inverted_col_ else x,axis=0)
        
    def get_feature_names_out(self):
        return self.feature_names_out_        

#%% imputer
"""
Création d'un class qui prend un fonction qui est applicable avec apply sur les colone à imputer. création de 3 fonction aussi
"""

def impute_sample(chridseries,random_st = None):
        """
        
    
        fonction visant a être utiliser avec un apply pour imputer les données 
        manquantes en tirant au hasard entre wt et muté sur la base de la fréquence
        de mutation du MS sur la séries de patients séquencé.
        ----------
        chridseries : pd.Series
            statuts de mutation pour un MS sur les n patients. 0 pour wt, 1 pour
            muté, np.nan pour inconnu.
        random_st : int
            DESCRIPTION. The default is None.
    
        Returns
        -------
        chridseries : pd.Series
            statuts de mutation pour un MS sur les n patients. 0 pour wt, 1 pour muté.
    
        """
        nullindex = chridseries.isnull()
        chridseries.loc[nullindex] = chridseries.dropna().sample(nullindex.sum(),random_state = random_st).values
        return chridseries
    
def impute_mean(chridseries):
        nullindex = chridseries.isnull()
        chridseries.loc[nullindex] = chridseries.dropna().mean()
        return chridseries   
    
def impute_most_frequent(chridseries):
        nullindex = chridseries.isnull()
        chridseries.loc[nullindex] = chridseries.dropna().median()
        return chridseries   
    
class impute_rnd(BaseEstimator, TransformerMixin):
    
    def __init__(self,applied_method=impute_sample):
        self.applied_method = applied_method
        return None
    



    
        
    def fit(self, X, y=None):
        self.fitted_status = "fitted"
        return self
        
    def transform(self,X,y=None):

        X.apply(self.applied_method)
        return X




#%% Cox_score function

# def fit_and_score_features_old(X, y,alpha=0.00001):
#     n_features = X.shape[1]
#     scores = np.empty(n_features)
#     m = CoxPHSurvivalAnalysis(alpha=alpha)
#     #suppress warnings
#     warnings.filterwarnings('ignore')
#     for j in range(n_features):
#         Xj = X[:, j:j+1]
#         try :
#             m.fit(X,y)
#         except:
#             continue
#         scores[j] = m.score(X,y)
    
#     warnings.filterwarnings('default')
#     return scores


def fit_and_score_features(X, y,alpha=0.00001):
    """
    https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html
    calculate the Cox score for each feature of a given X,y sksurv data

    Parameters
    ----------
    X : (array-like, shape = (n_samples, n_features)) 
        features matrix, rows <=> patients.
    y : (structured array, shape = (n_samples,))
        A structured array containing the binary event indicator as first field, and time of event or time of censoring as second field..

    Returns
    -------
    scores : (np.arrays, shape = (n_features))
        for each feature return Cox score.

    """

    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis(alpha=alpha)
    #suppress warnings
    warnings.filterwarnings('ignore')
    for j in range(n_features):
        Xj = X[:, j:j+1]
        nabool = ~np.isnan(Xj)
        try :
            m.fit(Xj[~np.isnan(Xj)], y[np.argwhere(nabool)[:,0]])
        except:
            continue
        scores[j] = m.score(Xj[~np.isnan(Xj)], y[np.argwhere(nabool)[:,0]])
    
    warnings.filterwarnings('default')
    return scores




# class bio_corex_pipeline(BaseEstimator,ce.Corex):
#     """
#     Class pour adapté bio_corex au pipline  https://github.com/gregversteeg/bio_corex/
#     """
#     def __init__(self,n_hidden=10,dim_hidden=2, marginal_description='discrete',smooth_marginals=False,n_cpu=1):
#         ce.Corex.__init__(self,n_hidden=n_hidden,dim_hidden=dim_hidden, marginal_description=marginal_description,smooth_marginals=smooth_marginals,n_cpu=n_cpu)
        
#     def fit(self,X,y=None):
        
#         super().fit(X.replace(np.NaN, -1).astype(int).values)
#         if isinstance(X, (pd.DataFrame, pd.Series)):
#             self.features = list(X.columns)
#         else:
#             self.features = list(range(X.shape[1]))
            
#         return self
    
#     def transform(self, X,y=None):
#         """Reduce dimensionality using CorEx https://github.com/gregversteeg/bio_corex/
        
#         Parameters
#         ----------
#         X : {array-like, sparse matrix}, shape (n_samples, n_features)
#             The input data to complete.
#         """
#         X2 = super().transform(X.replace(np.NaN, -1).astype(int).values)
#         if isinstance(X, (pd.DataFrame, pd.Series)):
#             return pd.DataFrame(
#                 data=X2, 
                
#                 index=X.index
#             )
#         else:
#             return pd.DataFrame(
#                 data=X2, 
#                 index=X.index
#             )
#     def fit_transform(self,X,y):
#         X2 = super().fit_transform(X.replace(np.NaN, -1).astype(int).values)
#         if isinstance(X, (pd.DataFrame, pd.Series)):
#             return pd.DataFrame(
#                 data=X2, 
                
#                 index=X.index
#             )
#         else:
#             return pd.DataFrame(
#                 data=X2, 
#                 index=X.index
#             )

#%%
# https://lms.fun-mooc.fr/courses/course-v1:inria+41026+session02/courseware/6b3b2cc0a9054b9492d498308f22ae6d/797f88079bcd491a8a651eaffca1ca80/

class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)



#%%

import numpy as np
from collections import Counter

def entropy(column):
    """
    Calcule l'entropie d'une colonne dans un DataFrame.

    Parameters:
        column (pandas.Series): La colonne pour laquelle calculer l'entropie.

    Returns:
        float: L'entropie de la colonne.
    """
    column = column.dropna()
    total_count = len(column)
    value_counts = Counter(column)
    entropy_value = 0.0

    for count in value_counts.values():
        probability = count / total_count
        entropy_value -= probability * np.log2(probability)

    return entropy_value

class EntropyFtSel(BaseEstimator, TransformerMixin):
    """
    Select feature with a minimum and maximum of mutation
    
    Parameters
    ---
    mini_counts = threshold value of selection. keep MS with more than mini_count mutation
    and with less than cohorte size - mini_count mutation
    """
    # initializer 
    def __init__(self, thr_entropy=0,entropy_fun=entropy):
        self.thr_entropy = thr_entropy
        self.entropy_fun = entropy_fun
        return None
        # save the features list internally in the class
        
    def fit(self, X, y=None):
        self.entropy = X.apply(self.entropy_fun,axis=0)
        self.feature_names_in_ = X.columns
        return self
    def transform(self, X, y = None):
        check_is_fitted(self,['entropy'])
        # return the dataframe with the specified features
        X2 = X.loc[:,self.entropy>self.thr_entropy]
        self.feature_names_out_ = X2.columns
        return X2
    
    def get_feature_names_out(self):
        return self.feature_names_out_




def minimum_nb_samples_with_value(counts_df,smallestGroupSize=3,thr=10,drop__col=0):

    
    # Identify the columns to drop from counts_df based on sum thresholds and non gene col
    if drop__col:
        col_to_drop = ["__no_feature", "__ambiguous", "__too_low_aQual", "__not_aligned", "__alignment_not_unique"]
    
        counts_df = counts_df.drop(col_to_drop, axis=1)
    # Sélectionne les gènes avec au moins "smallestGroupSize" échantillons ayant une expression supérieure ou égale à thr.
    gene_to_keep = counts_df.apply(lambda row: (row >= thr).sum(), axis=0) >= smallestGroupSize
    counts_df = counts_df.loc[:,gene_to_keep]

    return counts_df
    for count in value_counts.values():
        probability = count / total_count
        entropy_value -= probability * np.log2(probability)

    return entropy_value


