from config import directory_config,model_config
from utils import get_data,load_object
from preprocessing import run_preprocessing
from features import build_modelling_dataset
from algorithms import train_model

# required_columns  = ['ITM_CD','ITM_KEY','LS'] 
# modes = ['count','tf','tfidf']


dataset = get_data(directory_config,model_config)
if model_config['preprocessing']:
    dataset = run_preprocessing(dataset)
if model_config['generate_features']:
    dataset,feature_names = build_modelling_dataset(dataset = dataset,mode= model_config['mode'], total_features = model_config['total_features'])
else:
    feature_names = load_object(directory_config['root_dir'],directory_config['features_list_dir'],directory_config['features_list_object_name'] + ".pkl")
dataset = dataset[:5000]
final_clf,train_with_predictions,test_with_predictions = train_model(dataset,feature_names)


'''TO DO 
3. Create another virtual environment to check code
'''




