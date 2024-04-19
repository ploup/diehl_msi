import pandas as pd
# convert treatment to num : mono = 0 combo=1


#%% filt NaN
def filtnan(tabmut,filtcoef=0.05): 
    #drop les colonnes ayant plus que (coeff * nb de ligne) de nan (0 très strigent, 1 pas du tout)
    # Autrement dit, garde les colonne avec un proportion de NA inférieur à filtcoef
    filt_nan = len(tabmut)*filtcoef
    return tabmut.loc[:,(tabmut.isna().sum() <= filt_nan)] 


def clean_NA_duplicate_input(df_X,filtcoef=0.05,dup=True,n_estimators=50,max_iter=10):
    
    from sklearn.experimental import enable_iterative_imputer  # noqa

    from sklearn.impute import IterativeImputer
    from sklearn.ensemble import ExtraTreesRegressor
    N = df_X.shape[1]
    df_X = filtnan(df_X,filtcoef=filtcoef)
    print("The raw_dataset contains {0} feature (MS) with proportion of NA > {2} over {1} features".format(N-df_X.shape[1], N,filtcoef)) #0 null values
    
    if dup :
        # Removing duplicates if there exist
        N_dupli = sum(df_X.T.duplicated(keep='first'))
        df_X = df_X.T.drop_duplicates(keep='first').T
        print("The raw_dataset contains {} duplicates".format(N_dupli))

    # Number of samples in the dataset
    
    # Iterative imputation
    ## ExtraTree
    imp_tree = IterativeImputer(random_state=0,estimator=ExtraTreesRegressor(n_estimators=n_estimators, random_state=0),max_iter=max_iter)

    df_X= pd.DataFrame(imp_tree.fit_transform(df_X),columns=df_X.columns,index=df_X.index)
    return df_X