import numpy as np
# import statsmodels.api as sm
import pandas as pd
from scipy.stats import chi2
from matplotlib import pyplot as plt
from numpy.linalg import inv

### PROCéDURES FDR

# trouve la dernière occurence d'un élément
def find_last(lst, sought_elt):
    for r_idx, elt in enumerate(reversed(lst)):
        if elt == sought_elt:
            return len(lst) - 1 - r_idx
        
        
        
# effectue procedure de Benjamini-Hochberg au niveau alpha, lst étant la liste des pvalues ordonnées de manière croissante
# renvoie la p-value critique

def BH(lst,alpha):
    m = len(lst)
    seuils = [(alpha/m)*i for i in range(1,m+1)]
    boolean = [lst[i] <= seuils[i] for i in range(0,m)]
    index_crit = find_last(boolean,1)
    
    return lst[index_crit]



# effectue procedure de Benjamini-Yekutieli au niveau alpha, lst étant la liste des pvalues ordonnées de manière croissante
# renvoie la p-value critique

def BY(lst,alpha):
    m = len(lst)
    Hm = sum([1.0/i for i in range(1,m+1)])
    seuils = [(alpha/(m*Hm))*i for i in range(1,m+1)]
    boolean = [lst[i] <= seuils[i] for i in range(0,m)]
    index_crit = find_last(boolean,1)
    
    return lst[index_crit]



# effectue procedure de controle de la FDR rejetant plus d'hypothèses que celle de Benjamini-Yekutieli lorsque le nombre de
# positifs est grand 
# renvoie la p-value critique

def FDR3(lst,alpha):
    m = len(lst)
    seuils = [(alpha/(2*m**2))*i*(i+1) for i in range(1,m+1)]
    boolean = [lst[i] <= seuils[i] for i in range(0,m)]
    index_crit = find_last(boolean,1)
    
    return lst[index_crit]