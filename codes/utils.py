import pandas as pd
import pickle
import os,shutil

def get_dataset(root_dir,raw_data_dir,filename):
    data_frame = pd.read_csv(filepath_or_buffer= root_dir + raw_data_dir + filename,low_memory=False)
    return data_frame


def get_data(directory_config,model_config):
    if model_config['preprocessing']:
        dataset = get_dataset(directory_config['root_dir'],directory_config['raw_data_dir'],directory_config['filename'] + '.csv')
    else:
        if model_config['generate_features']:
            dataset = get_dataset(directory_config['root_dir'],directory_config['cleaned_data_dir'],directory_config['filename'] + '_cleaned' + '.csv')
        else:
            dataset = get_dataset(directory_config['root_dir'],directory_config['modelling_data_dir'],directory_config['filename'] + '_' + directory_config['features_list_object_name'] + '.csv')
    return dataset


def save_object(obj,root_directory,folder,file_name):
    with open(root_directory + folder + file_name + ".pkl", 'wb') as f:
        pickle.dump(obj, f)


def load_object(root_directory,folder,file_name):
    with open(root_directory + folder + file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def clear_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
                os.mkdir(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def copy_files(root_dir,folder,filename,target_location,target_folder):
    original = root_dir + folder + filename 
    dst = root_dir + target_location + target_folder 
    target = os.path.join(dst, os.path.basename(original))
    shutil.copyfile(original, target)


