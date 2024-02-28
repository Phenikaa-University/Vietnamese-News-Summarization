import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split as tts
import torch
import collections 
from collections import Counter
from torch import nn
from torch.utils.data import Dataset, DataLoader
from bs4 import BeautifulSoup 
import math

df = pd.read_csv("dataset/final/news-mds-extractive.csv")
data = df[['extractive_summary', 'summary']]
data.drop_duplicates(subset=['extractive_summary'], inplace=True)

# Preprocessing
def text_cleaner(text):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)

    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    long_words=[]
    text = " ".join(long_words).strip()
    def no_space(word, prev_word):
        return word in set(',!"";.''?') and prev_word!=" "
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    text = ''.join(out)
    return text

data['cleaned_text'] = data['extractive_summary'].apply(text_cleaner)
data['cleaned_summary'] = data['summary'].apply(text_cleaner)
# this step is to remove all rows that have a blank summary
# data["cleaned_summary"].replace('', np.nan, inplace=True)
# data.dropna(axis=0, inplace=True)

max_len_text=100 
max_len_summary=10

x_train,x_test,y_train,y_test = tts(data['cleaned_text'],data['cleaned_summary'],test_size=0.1, shuffle=True)

