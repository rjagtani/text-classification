directory_config = {
        'root_dir' : "C:\\Users\\Rohit Jagtani\\Desktop\\Projects\\text_classification",
        'raw_data_dir' : "\\data\\raw\\",
        'cleaned_data_dir' : "\\data\\cleaned\\",
        'modelling_data_dir' : "\\data\\modelling_dataset\\",
        'filename' : "dataset_all_labels",
        'features_list_dir' : "\\objects\\experiments\\features\\",
        'features_list_object_name' : "features_top200", 
        'transformer_dir' : "\\objects\\experiments\\transformer\\",
        'transformer_object_name' : "transformer_count_top200", 
        'model_object_dir' : "\\objects\\experiments\\model_object\\",
        'model_object_name' : "classifier_rf_v1",
        'train_predictions_dir' : "\\predictions\\experiments\\train\\",
        'test_predictions_dir' : "\\predictions\\experiments\\test\\",
        'final_objects_dir' : "\\objects\\final\\",
        'final_predictions_dir' : "\\predictions\\final\\"
}
        
model_config = {        
        'mode' : 'count',
        'total_features' : 200,
        'preprocessing' : True,
        'generate_features' : True
}