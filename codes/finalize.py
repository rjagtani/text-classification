from config import directory_config
from utils import clear_contents,copy_files

final_object_for_scoring = {
        'features_list_object_name' : "features_top200", 
        'transformer_object_name' : "transformer_count_top200", 
        'model_object_name' : "classifier_rf_v1"
}

clear_contents(folder = directory_config['root_dir'] + directory_config['final_objects_dir'])
copy_files(directory_config['root_dir'],directory_config['features_list_dir'],directory_config['features_list_object_name'] + ".pkl",directory_config['final_objects_dir'],"features")
copy_files(directory_config['root_dir'],directory_config['transformer_dir'],directory_config['transformer_object_name'] + ".pkl",directory_config['final_objects_dir'],"transformer")
copy_files(directory_config['root_dir'],directory_config['model_object_dir'],directory_config['model_object_name'] + ".pkl",directory_config['final_objects_dir'],"model_object")


clear_contents(folder = directory_config['root_dir'] + directory_config['final_predictions_dir'])
copy_files(directory_config['root_dir'],directory_config['train_predictions_dir'],directory_config['model_object_name'] + "_train_predictions.csv",directory_config['final_predictions_dir'],"train")
copy_files(directory_config['root_dir'],directory_config['test_predictions_dir'],directory_config['model_object_name'] + "_test_predictions.csv",directory_config['final_predictions_dir'],"test")
print("DONE")
    

