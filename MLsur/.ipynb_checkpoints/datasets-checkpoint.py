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