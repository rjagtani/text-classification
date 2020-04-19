import pandas as pd
from config import directory_config

def convert_columns_to_str(dataset,list_of_cols):
    dataset = dataset[list_of_cols].astype('str')
    return dataset

def get_duplicate_item_keys(dataset):
    agg_dict = {'ITM_CD' : ['min'],'LS':['nunique','min']}
    all_df_labels = dataset.groupby(['ITM_KEY'],as_index=False).agg(agg_dict)
    all_df_labels.columns = ['ITM_KEY','ITM_CD','total_labels','y']
    return all_df_labels

def remove_duplicate_item_keys(dataset):
    single_label_item_descriptions = dataset.ITM_KEY[dataset.total_labels==1].tolist()
    single_label_df = dataset[dataset.ITM_KEY.isin(single_label_item_descriptions)]
    del single_label_df['total_labels']
    single_label_df = single_label_df.reset_index(drop=True)
    return single_label_df

def run_preprocessing(dataset):
    dataset = convert_columns_to_str(dataset,['ITM_CD','ITM_KEY','LS'])
    dataset = get_duplicate_item_keys(dataset)
    dataset = remove_duplicate_item_keys(dataset)
    dataset.to_csv(directory_config['root_dir']  + directory_config['cleaned_data_dir'] + directory_config['filename'] + '_cleaned' + '.csv',index=False)
    return dataset
