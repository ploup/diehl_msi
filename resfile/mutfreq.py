import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import os

from collections import defaultdict



def create_boxplot_with_annotations(df, title=None, output_path=None, base_name=None):
    """
    Create a box plot of MS mutfreq in a cohort vs its length with annotations for the number 
    of points and save it as a PNG and/or HTML file.

    Args:
        df (DataFrame): The input DataFrame containing the data.
        title (str, optional): The title of the graph. Defaults to None.
        output_path (str, optional): The path to the directory where the files should be saved. Defaults to None.
        base_name (str, optional): The base name for the saved files. Defaults to None.

    Returns:
        fig: The generated Plotly figure.
    """
    # Filter the rows based on the condition
    filtered_df = df[df['nb_presence'] >= df['nb_presence'].max() / 2]
    hover_data = ["Gene.refGene", "Func.refGene"]

    # Create the graph using Plotly Express
    fig = px.box(filtered_df, x="length", y="norm", hover_data=hover_data)

    # Calculate and add annotations for the number of points
    counts = filtered_df.groupby("length")["norm"].count().reset_index(name='count')
    for index, row in counts.iterrows():
        fig.add_annotation(
            x=row['length'], y=1.05, text=f"N={row['count']}", showarrow=False, font=dict(size=10)
        )

    # Add the line connecting the means
    mean_values = filtered_df.groupby("length")["norm"].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=mean_values["length"],
            y=mean_values["norm"],
            mode="lines",
            name="Moyenne",
            line=dict(color="red")
        )
    )

    # Modify the graph title
    if title is not None:
        fig.update_layout(title=title)

    # Save as PNG if output_path is specified
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if base_name is None:
            base_name = title
        if base_name is None:
            base_name = "figure"
        png_path = f"{output_path}/{base_name}.png"
        fig.write_image(png_path)

    # Save as HTML if output_path is specified
    if output_path is not None:
        if base_name is None:
            base_name = title
        if base_name is None:
            base_name = "figure"
        html_path = f"{output_path}/{base_name}.html"
        fig.write_html(html_path)

    # Display the graph
    fig.update_layout(height=500)
    fig.show()
    return fig

def read_res_mutfreq_annotated(files, covlim=10, delthreshold=10, include_insertion=False):
    """
    Reads mutation frequency annotated files and returns a DataFrame with mutation information.

    Args:
        files (list): A list of file paths to the mutation frequency annotated files.
        covlim (int, optional): The minimum depth of sequencing required for a mutation to be considered. Defaults to 10.
        delthreshold (int, optional): The delta ratio threshold for considering a deletion mutation. Defaults to 10.
        include_insertion (bool, optional): Whether to include insertion mutations. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing mutation information, including the number of mutations, the repeated nucleotide, the length of the mutation, the number of patients sequenced, and the mean depth on normal and tumor tissue.

    """
    dico_mut_count = defaultdict(lambda: [0, '', 0, 0, 0, 0, '', ''])

    if include_insertion:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '1', '2']
    else:
        indel_len = ['-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1']

    for path_res in files:
        df = pd.read_table(path_res, sep='\t', index_col=0)
        df = df[(df['sum_Tum'] >= covlim) & (df['sum_Nor'] >= covlim)]
        df.rename({'Func.refGene': 'Func_refGene', 'Gene.refGene': 'Gene_refGene'}, axis=1, inplace=True)
        df['Mutated'] = (df[indel_len] > delthreshold).any(axis=1).astype(int)

        for row_tuple in df.itertuples():
            chrid = row_tuple.Index
            mut = row_tuple.Mutated
            sum_norm = row_tuple.sum_Nor
            sum_tum = row_tuple.sum_Tum
            func_refgene = row_tuple.Func_refGene
            gene_refgene = row_tuple.Gene_refGene

            dico_mut_count[chrid][0] += mut
            dico_mut_count[chrid][1] = row_tuple.nucleo
            dico_mut_count[chrid][2] = row_tuple.length
            dico_mut_count[chrid][3] += 1
            dico_mut_count[chrid][4] += sum_norm
            dico_mut_count[chrid][5] += sum_tum
            dico_mut_count[chrid][6] = func_refgene
            dico_mut_count[chrid][7] = gene_refgene

    df_mut_count = pd.DataFrame.from_dict(
        dico_mut_count,
        orient='index',
        columns=['nb_mut', 'nucleo', 'length', 'nb_presence', 'sum_norm_mean', 'sum_tum_mean', 'Func.refGene', 'Gene.refGene']
    )

    df_mut_count['norm'] = df_mut_count['nb_mut'].div(df_mut_count['nb_presence'])
    df_mut_count['sum_norm_mean'] = df_mut_count['sum_norm_mean'].div(df_mut_count['nb_presence'])
    df_mut_count['sum_tum_mean'] = df_mut_count['sum_tum_mean'].div(df_mut_count['nb_presence'])

    return df_mut_count


def read_res_mutfreq(files, covlim=10,delthreshold=10, bed_filt=False, bed_set=set(),include_insertion=False):
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


def read_res_old_MSIcare(files,MSIcaredf=None,covlim=20,bed_filt=False,bed_set=set(),include_insertion=False,delthreshold=10):
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
                mutated_index = df[(df[indel_len] > delthreshold).any(axis=1)].index
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
                                   