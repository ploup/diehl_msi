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



def upsampling(Xtrain,Etrain,size_resample=1000):
    
    # Separate majority and minority classes
    E_train_majority = Etrain[Etrain==0]
    E_train_minority = Etrain[Etrain==1]

    # Upsample minority class
    E_train_minority_upsampled = resample(E_train_minority, 
                                     replace=True,     # sample with replacement
                                     n_samples=size_resample,    # to match majority class
                                     random_state=123) # reproducible results

    # Upsample majority class
    E_train_majority_upsampled = resample(E_train_majority, 
                                     replace=True,     # sample with replacement
                                     n_samples=size_resample,    # to match majority class
                                     random_state=123) # reproducible results

    # Combine majority class with upsampled minority class
    E_train_upsampled = pd.concat([E_train_majority_upsampled, E_train_minority_upsampled])

    
    X_train_upsampled = Xtrain.loc[E_train_upsampled.index]


    return X_train_upsampled, E_train_upsampled

def censore_survivaldata(df:pd.DataFrame, duration_col : str, event_col : str, time : int) -> pd.DataFrame:
    """
    Function that censors clinical data at 60 months
    if duration_col > time, then event_col = 0 and duration_col = time
    """

    df.loc[(df[duration_col].astype(float) > time), (event_col,)] = 0
    df.loc[(df[duration_col].astype(float) > time), duration_col] = time

    return df
    