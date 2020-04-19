import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from config import directory_config
#from nltk.stem.snowball import SnowballStemmer
nltk.download('wordnet',quiet=True, raise_on_error=True)
nltk.download('stopwords', quiet=True, raise_on_error=True)
import pickle
from utils import save_object


def custom_tokenizer(doc):
    pattern = re.compile(r'\b\w\w+\b')
    doc1 = re.findall(pattern,doc)
    #doc2 = [SnowballStemmer(language='english').stem(term.strip()) for term in doc1] 
    doc2 = [WordNetLemmatizer().lemmatize(term) for term in doc1]
    return doc2

def build_vectorizer(mode,total_features):
    vectorizer = None
    tokenized_stop_words = custom_tokenizer(' '.join(nltk.corpus.stopwords.words('english')))
    if mode == 'count':
        vectorizer = CountVectorizer(analyzer = 'word',stop_words = tokenized_stop_words,tokenizer = custom_tokenizer,max_features=total_features)
    elif mode == 'tf':
        vectorizer = TfidfVectorizer(analyzer = 'word',stop_words = tokenized_stop_words,tokenizer = custom_tokenizer,max_features=total_features,use_idf=False, norm='l2')
    elif mode == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer = 'word',stop_words = tokenized_stop_words,tokenizer = custom_tokenizer,max_features=total_features)
    else:
        raise ValueError('Mode should be either count or tfidf')
    return vectorizer

def build_modelling_dataset(dataset,mode,total_features):
    vectorizer = build_vectorizer(mode,total_features)
    output = vectorizer.fit_transform(dataset['ITM_KEY'].tolist())
    save_object(vectorizer,directory_config['root_dir'],directory_config['transformer_dir'],directory_config['transformer_object_name'])
    output_array = output.toarray()
    output_df = pd.DataFrame(output_array)
    feature_names = vectorizer.get_feature_names()
    output_df.columns = feature_names
    save_object(feature_names,directory_config['root_dir'],directory_config['features_list_dir'],directory_config['features_list_object_name'])
    modelling_dataset = pd.concat([output_df,dataset],axis=1)
    modelling_dataset.to_csv(directory_config['root_dir']  + directory_config['modelling_data_dir'] + directory_config['filename'] + '_' + directory_config['features_list_object_name'] + '.csv',index=False)
    return modelling_dataset,feature_names





