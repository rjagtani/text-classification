text_classification
==============================

Multiclass Text Classification

Goal : The aim of the project is to use apparel description to categorize them into 4 apparel categories; however due to the modularized nature of the code it can be used for other text classification tasks with some minor tweaking 

Flow: Fetch Data --> Preprocessing(optional) --> Feature Generation (Optional) --> Train Models (Run multiple experiments by changing filenames in config) --> Finalize Model --> Model Scoring

To replicate the project: Change root directory in config, create virtual environment in python, run 'pip install -r requirements.txt and then main.py

File Description and purpose:

Config.py: Contains model config settings and dictionary to store directories and object names

Preprocessing.py: Functions to clean the raw data; specific to the raw data received and the desired output; change for your specific task

Features.py: Uses nltk to build custom tokenizer,vectorizer; option to play around and try other configurations such as count,tf,tf-idf (check config.py),saves transformer for model scoring

Algorithm.py: Used for model training; currently has RandomForest with GridSearchCV, however other models can be added as functions or as part of other experiments; saves the best model from Grid Search results to 'objects/experiments' and predictions to 'predictions/experiments'

finalize.py: Has a dictionary for the objects (Feature names,transformer,model) that have been finalized. Copies the selected objects from 'objects/experiments' to 'objects/experiments' for scoring

scoring.py: Takes data without labels, does preprocessing(optional and data specific), applies transformers, generates features and scores using the final model. The final predictions are saved to 'scoring/predictions'

utils.py: Contains helper functions





