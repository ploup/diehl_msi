import pandas as pd


def import_CPTAC_surv():

    CPTAC_surv = pd.read_table("./CPTAC_clindata_gdc/coad_cptac_2019_clinical_data.tsv",usecols=['Patient ID',"Overall Survival Status","Overall Survival (Months)"],index_col=0)
    CPTAC_surv =  CPTAC_surv.rename(columns={"Overall Survival Status":"OS","Overall Survival (Months)":"OS.time"})
    CPTAC_surv = CPTAC_surv.replace("0:LIVING",0)
    CPTAC_surv = CPTAC_surv.replace("1:DECEASED",1)
    
    return CPTAC_surv

def import_TIMSI_surv():
    TIMSI_surv = pd.read_csv('/home/aurelien/Workspace/CMSI/2.Survival_analysis/clinical_data/Clin_data_cohorts/TIMSIclean.csv',index_col=0)[['OVERALL_SURVIVAL.1','OS_STATUS.1']].dropna()
    TIMSI_surv.columns = ["OS.time","OS"]
    
    return TIMSI_surv


def import_TCGA_surv():
    TCGA_surv = pd.read_excel('/home/aurelien/Workspace/CMSI_Clean/SVC_Randomforest/TCGA_clinical_data.xlsx',index_col=1)[['OS','OS.time']]
    TCGA_surv['OS.time']= TCGA_surv['OS.time'] * 0.0328767
    
    
    return TCGA_surv


def KM_median_followup(df_surv : pd.DataFrame, duration_col : str, event_col : str, cohort_name : str) -> KaplanMeierFitter:
    """
    Generate a Kaplan-Meier follow-up survival curve plot for a given DataFrame.

    Parameters:
    - df_surv : pd.DataFrame : The DataFrame containing the survival data.
    - duration_col : str : The column name in df_surv representing the duration.
    - event_col : str : The column name in df_surv representing the event.
    - cohort_name : str : The name of the cohort for labeling purposes.

    Returns:
    - KaplanMeierFitter : The KaplanMeierFitter object containing the fit survival curve.
    """
    # Make a copy of the DataFrame to avoid modifying the original one
    df_surv_copy = df_surv.copy()
    
    df_surv_copy[event_col] = np.abs(df_surv_copy[event_col]-1)
    # drop line wher event_col or duration_col is NA
    df_surv_copy = df_surv_copy.dropna(subset=[event_col, duration_col])

    kmf = KaplanMeierFitter()
    kmf.fit(df_surv_copy[duration_col], df_surv_copy[event_col], label=cohort_name)
    
    fig = plt.figure()
    plt.axhline(0.5, color="r", linestyle='dashdot')
    kmf.plot()

    return kmf

def plot_multiple_km(list_of_km):
    """
    Plot multiple Kaplan-Meier follow-up curves on the same graph.

    Parameters:
    - list_of_km: A list of KaplanMeierFitter objects to plot.

    Returns:
    - fig: The generated figure.
    - ax: The axes object containing the plot.
    """
    fig, ax = plt.subplots()
    ax.axhline(0.5, color="r", linestyle='--', label='50% Follow-up')
    
    for km in list_of_km:
        km.survival_function_.plot(ax=ax)
    
    # Set the x-axis to be in months
    ax.set_xlabel('Months')
    # Set the y-axis to represent follow-up
    ax.set_ylabel('Follow-up')
    # Invert the y-axis to have 1 represent no follow-up and 0 represent full follow-up
    ax.invert_yaxis()
    # Add a legend
    ax.legend()
    
    return fig, ax