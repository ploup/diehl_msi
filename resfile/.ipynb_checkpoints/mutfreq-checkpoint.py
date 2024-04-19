import pandas as pd



from collections import defaultdict

def read_res_mutfreq(files, covlim=10, bed_filt=False, bed_set=set(),include_insertion=False):
    # create an empty dictionnary which return for each MS (key) the number of mutation, the repeated nucleotide, the length, how many patient are sequenced, the mean depth on the normal and tumor tissue

    dico_mut_count = defaultdict(lambda: [0, '', 0, 0,   0, 0])# Using defaultdict with default values for each key

    if include_insertion:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2']

    else : 
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1']


    # iterate over the input res files
    for path_res in files:
        
        # read res file
        df = pd.read_table(path_res,sep='\t',index_col=0)
        
        #keep only MS respecting the minimum depth of sequencing
        df = df[(df['sum_Tum']>=covlim) & (df['sum_Nor']>=covlim)]
        
        # Consider a MS mutated if the delta ratio is > 10 in any deletion length
        df['Mutated'] = (df[indel_len] > delthreshold).any(axis=1).astype(int)

        
        # iterate over MS (each line of the res file)
        for row_tuple in df.itertuples():
            # Extract value of interest
            chrid = row_tuple.Index # position
            mut = row_tuple.Mutated # Mutation (0 or 1)
            sum_norm = row_tuple.sum_Nor # depth of norm
            sum_tum = row_tuple.sum_Tum # depth of tumor
            
            # Add those values to the dict
            dico_mut_count[chrid][0] += mut
            dico_mut_count[chrid][1] = row_tuple.nucleo
            dico_mut_count[chrid][2] = row_tuple.length
            dico_mut_count[chrid][3] += 1
            dico_mut_count[chrid][4] += sum_norm
            dico_mut_count[chrid][5] += sum_tum


    # convert the dict into a dataframe
    df_mut_count = pd.DataFrame.from_dict(dico_mut_count,orient='index',columns=['nb_mut','nucleo','length','nb_presence','sum_norm_mean','sum_tum_mean'])

    # calculate the mutation frequency as the (number of patient mutated on the MS)/(number of patient sequenced on the MS)
    df_mut_count['norm'] = df_mut_count['nb_mut'].div(df_mut_count['nb_presence'])
    
    # calculate the mean depth
    df_mut_count['sum_norm_mean'] = df_mut_count['sum_norm_mean'].div(df_mut_count['nb_presence'])
    df_mut_count['sum_tum_mean'] = df_mut_count['sum_tum_mean'].div(df_mut_count['nb_presence'])
            
    # if bed_filt: #if bed filter given only consider position in the bed
    #     df_mut_count = df_mut_count.reindex(bed_set).dropna()

    return df_mut_count


def read_res_old_MSIcare(files,MSIcaredf=None,covlim=20,bed_filt=False,bed_set=set()(),include_insertion=False):
        dico_mut_count = {}
            if include_insertion:
                indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2']
        
            else : 
                indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1']
        
        for path_res in tqdm.tqdm(files):
            sample_name = path_res.split('/')[-1][:-4]
            if MSIcaredf.loc[sample_name]['diag'] == 'MSI':
                df = pd.read_table(path_res,sep='\t',index_col=0)
                df = df[(df['sum_Tum']>=covlim) & (df['sum_Nor']>=covlim)]
                df['Mutated'] = 0
                mutated_index = df[(df[indel_len] > 10).any(axis=1)].index
                df.loc[mutated_index,'Mutated'] = 1
                
                for row_tuple in df.itertuples():
                    chrid = row_tuple.Index
                    mut = row_tuple.Mutated
                    sum_norm = row_tuple.sum_Nor
                    sum_tum = row_tuple.sum_Tum
                    
                    if chrid in dico_mut_count:
                        dico_mut_count[chrid][0] += mut 
                        dico_mut_count[chrid][3] += 1
                        dico_mut_count[chrid][4] += sum_norm
                        dico_mut_count[chrid][5] += sum_tum
                    else : 
                        dico_mut_count[chrid]= [mut,row_tuple.nucleo,row_tuple.length,1,sum_norm,sum_tum]

                df_mut_count = pd.DataFrame.from_dict(dico_mut_count,orient='index',columns=['nb_mut','nucleo','length','nb_presence','sum_norm_mean','sum_tum_mean'])

                        

            
        df_mut_count = pd.DataFrame.from_dict(dico_mut_count,orient='index',columns=['nb_mut','nucleo','length','nb_presence','sum_norm_mean','sum_tum_mean'])

        df_mut_count['norm'] = df_mut_count['nb_mut'].div(df_mut_count['nb_presence'])
        df_mut_count['sum_norm_mean'] = df_mut_count['sum_norm_mean'].div(df_mut_count['nb_presence'])
        df_mut_count['sum_tum_mean'] = df_mut_count['sum_tum_mean'].div(df_mut_count['nb_presence'])
        # if bed_filt:
        #     df_mut_count_MSS = df_mut_count_MSS.reindex(bed_set).dropna()
        #     df_mut_count_MSI = df_mut_count_MSI.reindex(bed_set).dropna()
        
        return df_mut_count              
                                   