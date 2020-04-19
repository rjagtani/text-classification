import pandas as pd
import os
from config import directory_config
from preprocessing import convert_columns_to_str
from utils import load_object

feature_names = load_object(directory_config['root_dir'],directory_config['final_objects_dir'] + "features\\",os.listdir(directory_config['root_dir'] + directory_config['final_objects_dir'] + "features")[0])
model_object = load_object(directory_config['root_dir'],directory_config['final_objects_dir'] + "model_object\\",os.listdir(directory_config['root_dir'] + directory_config['final_objects_dir'] + "model_object")[0])
transformer = load_object(directory_config['root_dir'],directory_config['final_objects_dir'] + "transformer\\",os.listdir(directory_config['root_dir'] + directory_config['final_objects_dir'] + "transformer")[0])

for filename in os.listdir(directory_config['root_dir'] + "\\scoring\\data"):
    file_name = filename.replace(".csv","")
    scoring_dataset = pd.read_csv(directory_config['root_dir'] + "\\scoring\\data\\" + filename)
    scoring_dataset = convert_columns_to_str(scoring_dataset,['ITM_CD','ITM_KEY'])
    scoring_dataset_vectorize = transformer.transform(scoring_dataset['ITM_KEY'].tolist())
    scoring_dataset_features = pd.DataFrame(scoring_dataset_vectorize.toarray())
    scoring_dataset_features.columns = feature_names
    scoring_dataset_features = pd.concat([scoring_dataset_features,scoring_dataset],axis=1)
    new_predictions = model_object.predict(scoring_dataset_features[feature_names].copy())
    scoring_dataset_features.loc[:,'predictions'] = new_predictions
    scoring_dataset_features.to_csv(directory_config['root_dir'] + "\\scoring\\predictions\\" + file_name + '_predictions.csv',index=False)


