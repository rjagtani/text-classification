import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,average_precision_score,f1_score,classification_report
from utils import save_object
from config import directory_config

def grid_search_random_forest(X_train,y_train,grid_search_parameters,cv,n_jobs):
    rf_clf = RandomForestClassifier(n_estimators=500,n_jobs=n_jobs)
    cv_clf = GridSearchCV(rf_clf,grid_search_parameters,n_jobs=n_jobs,cv=cv)
    print("GRID SEARCH STARTED")
    cv_clf.fit(X_train,y_train)
    cv_results_df = pd.DataFrame(cv_clf.cv_results_)
    print("best_params : " + str(cv_clf.best_params_))
    print("best_score : " + str(cv_clf.best_score_))
    best_clf = cv_clf.best_estimator_
    return best_clf
    

def model_evaluation(data_frame):
    print(classification_report(data_frame['y'],data_frame['predictions']))


def get_datasets_with_predictions(final_clf,X_train,X_test,feature_names):
    train_predictions = final_clf.predict(X_train.loc[:,feature_names].copy())
    X_train.loc[:,"predictions"] = train_predictions
    test_predictions = final_clf.predict(X_test.loc[:,feature_names].copy())
    X_test.loc[:,"predictions"] = test_predictions
    return X_train,X_test


def train_model(dataframe,feature_names):
    X = dataframe
    y= dataframe['y'].values.reshape(-1,)
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=11)
    grid_search_parameters = {'max_features':[15,20]}
    final_clf = grid_search_random_forest(X_train[feature_names].copy(),y_train,grid_search_parameters,cv=5,n_jobs=-1)
    save_object(final_clf,directory_config['root_dir'],directory_config['model_object_dir'],directory_config['model_object_name'])
    train_with_pred,test_with_pred = get_datasets_with_predictions(final_clf,X_train.copy(),X_test.copy(),feature_names)
    train_with_pred.to_csv(directory_config['root_dir']  + directory_config['train_predictions_dir'] + directory_config['model_object_name'] + '_train_predictions'  + '.csv',index=False)
    test_with_pred.to_csv(directory_config['root_dir']  + directory_config['test_predictions_dir'] + directory_config['model_object_name'] + '_test_predictions' + '.csv',index=False)
    print("Evaluation on train")
    model_evaluation(train_with_pred)
    print("Evaluation on test")
    model_evaluation(test_with_pred)
    return final_clf,train_with_pred,test_with_pred

