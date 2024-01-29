"""Create Preprocess class for preprocessing data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
import re

class Preprocess_News_dataset():
    def __init__(self, path, save_path):
        self.path = path
        self.save_path = save_path

    def read_data(self):
        # Read data
        df = pd.read_csv(self.path)
        # Combine Title and Contents columns
        df['text'] = df['Title'] + ' ' + df['Contents']
        # Extract data['text'] and data['Summary] columns to new dataframe
        data = pd.DataFrame()
        data['text'] = df['text']
        data['summary'] = df['Summary']
        lst_stopwords = self.create_stopwords()
        data = self.add_preprocessed_text(data, column="text", 
                                          punkt=True, lower=True, lst_stopwords=lst_stopwords)
        data = self.add_preprocessed_text(data, column="summary",
                                          punkt=True, lower=True, lst_stopwords=lst_stopwords)
        if self.save_path is not None:
            data.to_csv(f'{self.save_path}/preprocessed_data.csv', index=False)
            return data
        return data
    
    def create_stopwords(self, lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
        '''
        Creates a list of stopwords.
        :parameter
            :param lst_langs: list - ["english"]
            :param lst_add_words: list - list of new stopwords to add
            :param lst_keep_words: list - list words to keep (exclude from stopwords)
        :return
            stop_words: list of stop words
        '''  
        lst_stopwords = set()
        for lang in lst_langs:
            lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
        lst_stopwords = lst_stopwords.union(lst_add_words)
        lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
        return sorted(list(set(lst_stopwords)))
    
    
    def add_preprocessed_text(self, data, column, lst_regex=None, punkt=False, lower=False, lst_stopwords=None, remove_na=True):
        '''
        Adds a column of preprocessed text.
        :parameter
            :param dtf: dataframe - dtf with a text column
            :param column: string - name of column containing text
        :return
            : input dataframe with two new columns
        '''
        dtf = data.copy()

        ## apply preprocess
        dtf = dtf[ pd.notnull(dtf[column]) ]
        dtf[column+"_clean"] = dtf[column].apply(lambda x: self.utils_preprocess_text(x, lst_regex, punkt, lower, lst_stopwords))
        
        ## residuals
        dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
        if dtf["check"].min() == 0:
            print("--- found NAs ---")
            print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
            if remove_na is True:
                dtf = dtf[dtf["check"]>0] 
                
        return dtf.drop("check", axis=1)
    
    def utils_preprocess_text(self, txt, lst_regex=None, punkt=True, lower=True, lst_stopwords=None):
        '''
        Preprocess a string.
        :parameter
            :param txt: string - name of column containing text
            :param lst_resgex: list - list of regex to remove
            :param punkt: bool - if True removes punctuations and characters
            :param lower: bool - if True convert lowercase
            :param lst_stopwords: list - list of stopwords to remove
        :return
            cleaned text
        '''
        ## Regex (in case, before cleaning)
        if lst_regex is not None: 
            for regex in lst_regex:
                txt = re.sub(regex, '', txt)

        ## Clean 
        ### Remove special characters
        txt = re.sub(r'[0-9\.\,\!\@\#\$\%\^\&\*\?]+', '', str(txt))
        ### separate sentences with '. '
        txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
        ### remove punctuations and characters
        txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
        ### strip
        txt = " ".join([word.strip() for word in txt.split()])
        ### lowercase
        txt = txt.lower() if lower is True else txt
                
        ## Tokenize (convert from string to list)
        lst_txt = txt.split()
                    
        ## Stopwords
        if lst_stopwords is not None:
            lst_txt = [word for word in lst_txt if word not in lst_stopwords]
                
        ## Back to string
        txt = " ".join(lst_txt)
        return txt
    
# data_path = "data/Dataset_articles.csv"
# preprocess = Preprocess_News_dataset(data_path, save_path="data").read_data()