# %%
import warnings
warnings.filterwarnings("ignore")

# %%
## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for analysis
import re
import langdetect 
import nltk
import wordcloud
import contractions

# %%
## for sentiment
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

## for ner
import spacy
import collections

## for machine learning
from sklearn import preprocessing, model_selection, feature_extraction, feature_selection, metrics, manifold, naive_bayes, pipeline

## for deep learning
from tensorflow.keras import callbacks, models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

# %%
## for explainer
from lime import lime_text
import shap

## for W2V and textRank
import gensim
import gensim.downloader as gensim_api

## for bert/bart
import transformers

## for summarization
import rouge
import difflib

# %% [markdown]
# ## Import data

# %%
data_path = "../data/Dataset_articles.csv"

# %%
df = pd.read_csv(data_path)
df.head()

# %%
df.info()

# %%
# Combine Title and Content columns
df['text'] = df['Title'] + " " + df['Contents']
# Combine df['text'] and df['Summary'] columns to new dataframe
data = pd.DataFrame()
data['text'] = df['text']
data['summary'] = df['Summary']

# %%
data.head()

# %%
data.info()

# %%
# Check sample
i = 1
print("====== Full text ======")
print(data['text'][i])
print("\n====== Summary ======")
print(data['summary'][i])

# %% [markdown]
# ## Text cleaning

# %%
nltk.download('stopwords')
nltk.download('wordnet')

# %%
'''
Creates a list of stopwords.
:parameter
    :param lst_langs: list - ["english", "italian"]
    :param lst_add_words: list - list of new stopwords to add
    :param lst_keep_words: list - list words to keep (exclude from stopwords)
:return
    stop_words: list of stop words
'''      
def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))

# %%
'''
Preprocess a string.
:parameter
    :param txt: string - name of column containing text
    :param lst_regex: list - list of regex to remove
    :param punkt: bool - if True removes punctuations and characters
    :param lower: bool - if True convert lowercase
    :param lst_stopwords: list - list of stopwords to remove
:return
    cleaned text
'''
def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, lst_stopwords=None):
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

# %%
'''
Adds a column of preprocessed text.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    : input dataframe with two new columns
'''
def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, lst_stopwords=None, remove_na=True):
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, lst_stopwords))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)

# %%
data.head()

# %%
lst_stopwords = create_stopwords()
data = add_preprocessed_text(data, column="text", 
                            punkt=True, lower=True, lst_stopwords=lst_stopwords)
data = add_preprocessed_text(data, column="summary", 
                            punkt=True, lower=True, lst_stopwords=lst_stopwords)

# %%
data.head()

# %%
# check
print("--- Full text ---")
print(data["text_clean"][i])
print(" ")
print("--- Summary ---")
print(data["summary_clean"][i])

# %%
# Save data to preprocessed_data.csv
data.to_csv("../data/preprocess/preprocessed_data.csv", index=False)

# %% [markdown]
# ## Word Frequency

# %%
'''
Compute n-grams frequency with nltk tokenizer.
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''
def word_freq(corpus, ngrams=[1,2,3], top=10, figsize=(10,7), name="word_freq"):
    lst_tokens = nltk.tokenize.word_tokenize(corpus.str.cat(sep=" "))
    ngrams = [ngrams] if type(ngrams) is int else ngrams
    
    ## calculate
    dtf_freq = pd.DataFrame()
    for n in ngrams:
        dic_words_freq = nltk.FreqDist(nltk.ngrams(lst_tokens, n))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dtf_freq = pd.concat([dtf_freq, dtf_n])
    dtf_freq["word"] = dtf_freq["word"].apply(lambda x: " ".join(string for string in x) )
    dtf_freq = dtf_freq.sort_values(["ngrams","freq"], ascending=[True,False])
    
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=dtf_freq.groupby('ngrams')[["ngrams","freq","word"]].head(top))
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()
    plt.savefig(f"../figs/{name}.png")
    return dtf_freq

# %%
# Find most common words in text
data_freq_x = word_freq(corpus=data["text_clean"], ngrams=[1], top=30, figsize=(10,7), name="data_freq_x")

# %%
thres = 5
X_top_words = len(data_freq_x[data_freq_x["freq"]>thres])
X_top_words

# %%
# Find most common words in text
data_freq_y = word_freq(corpus=data["summary_clean"], ngrams=[1], top=30, figsize=(10,7), name="data_freq_y")

# %%
thres = 5
y_top_words = len(data_freq_y[data_freq_y["freq"]>thres])
y_top_words

# %%
'''
Compute different text length metrics.
:parameter
    :param dtf: dataframe - dtf with a text column
    :param column: string - name of column containing text
:return
    dtf: input dataframe with 2 new columns
'''
def add_text_length(data, column):
    dtf = data.copy()
    dtf['word_count'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))) )
    dtf['char_count'] = dtf[column].apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))) )
    dtf['sentence_count'] = dtf[column].apply(lambda x: len(nltk.sent_tokenize(str(x))) )
    dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
    dtf['avg_sentence_lenght'] = dtf['word_count'] / dtf['sentence_count']
    print(dtf[['char_count','word_count','sentence_count','avg_word_length','avg_sentence_lenght']].describe().T[["min","mean","max"]])
    return dtf

# %%
# Texts
X = add_text_length(data, "text_clean")

# %%
'''
Plot univariate and bivariate distributions.
'''
def plot_distributions(dtf, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5), name="plot_distributions"):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if dtf[x].nunique() <= max_cat:
            if top is None:
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(dtf[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

    ## bivariate
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=15)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="", ax=ax[1])
        ax[0].set(title="histogram")
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].set(title="density")
        ax[1].grid(True)
    plt.show()
    plt.savefig(f"../figs/{name}.png")

# %%
plot_distributions(X, x="word_count", figsize=(10,3), name="plot_distributions_word_count_x")

# %%
# Summaries
summaries = add_text_length(data, "summary_clean")

# %%
plot_distributions(summaries, x="word_count", max_cat=1, figsize=(10,3), name="plot_distributions_word_count_y")

# %%
data.head()

# %% [markdown]
# ## Preprocess

# %%
df_train = data.iloc[i+1:]
df_test = data.iloc[:i+1]
df_test

# %% [markdown]
# ## Baseline (Extractive: Text rank)

# %%
'''
Summarizes corpus with TextRank.
:parameter
    :param corpus: list - dtf["text"]
    :param ratio: length of the summary (ex. 20% of the text)
:return
    list of summaries
'''
def textrank(corpus, ratio=0.2):
    if type(corpus) is str:
        corpus = [corpus]
    lst_summaries = [gensim.summarization.summarize(txt, ratio=ratio) for txt in corpus]
    return lst_summaries

# %%
from nltk.tokenize import sent_tokenize, word_tokenize

def preprocess_text(text):
    sentences = sent_tokenize(text)
    sentences = [' '.join(word_tokenize(sent)) for sent in sentences]
    return ' '.join(sentences)

df_test["text"] = df_test["text"].apply(preprocess_text)
predicted = textrank(corpus=df_test["text"], ratio=40/1000)

# %%
df_test['text_clean'][0]

# %%
summary

# %%
predicted = textrank(corpus=df_test["text_clean"], ratio=0.2)

# %% [markdown]
# ## Model (Abstractive: Seq2Seq)

# %%
'''
Create a list of lists of grams with gensim:
    [ ["hi", "my", "name", "is", "Tom"], 
      ["what", "is", "yours"] ]
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
:return
    lst of lists of n-grams
'''
def utils_preprocess_ngrams(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[]):
    ## create list of n-grams
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [grams_join.join(lst_words[i:i + ngrams]) for i in range(0, len(lst_words), ngrams)]
        lst_corpus.append(lst_grams)
    
    ## detect common bi-grams and tri-grams
    if len(lst_ngrams_detectors) != 0:
        for detector in lst_ngrams_detectors:
            lst_corpus = list(detector[lst_corpus])
    return lst_corpus

# %%
'''
Transforms the corpus into an array of sequences of idx (tokenizer) with same length (padding).
:parameter
    :param corpus: list - dtf["text"]
    :param ngrams: num - ex. "new", "york"
    :param grams_join: string - "_" (new_york), " " (new york)
    :param lst_ngrams_detectors: list - [bigram and trigram models], if empty doesn't detect common n-grams
    :param fitted_tokenizer: keras tokenizer - if None it creates one with fit and transorm (train set), if given it transforms only (test set)
    :param top: num - if given the tokenizer keeps only top important words
    :param oov: string - how to encode words not in vocabulary (ex. "NAN")
    :param maxlen: num - dimensionality of the vectors, if None takes the max length in corpus
    :param padding: string - <PAD> token
:return
    If training: matrix of sequences, tokenizer, dic_vocabulary. Else matrix of sequences only.
'''
def text2seq(corpus, ngrams=1, grams_join=" ", lst_ngrams_detectors=[], fitted_tokenizer=None, top=None, oov=None, maxlen=None, padding="<PAD>"):    
    print("--- tokenization ---")
    
    ## detect common n-grams in corpus
    lst_corpus = utils_preprocess_ngrams(corpus, ngrams=ngrams, grams_join=grams_join, lst_ngrams_detectors=lst_ngrams_detectors)

    ## bow with keras to get text2tokens without creating the sparse matrix
    ### train
    if fitted_tokenizer is None:
        tokenizer = kprocessing.text.Tokenizer(num_words=top, lower=False, split=' ', char_level=False, oov_token=oov,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(lst_corpus)
        dic_vocabulary = {padding:0}
        words = tokenizer.word_index if top is None else dict(list(tokenizer.word_index.items())[0:top+1])
        dic_vocabulary.update(words)
        print(len(dic_vocabulary), "words")
    else:
        tokenizer = fitted_tokenizer
    ### transform
    lst_text2seq = tokenizer.texts_to_sequences(lst_corpus)

    ## padding sequence (from [1,2],[3,4,5,6] to [0,0,1,2],[3,4,5,6])
    print("--- padding to sequence ---")
    X = kprocessing.sequence.pad_sequences(lst_text2seq, maxlen=maxlen, padding="post", truncating="post")
    print(X.shape[0], "sequences of length", X.shape[1]) 

    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sequences Overview')
    plt.show()
    return {"X":X, "tokenizer":tokenizer, "dic_vocabulary":dic_vocabulary} if fitted_tokenizer is None else X

# %%
# Create X_train for seq2seq (sequences of tokens)
dic_seq = text2seq(corpus=df_train["text_clean"], top=X_top_words, maxlen=200)


# %%
X_train, X_tokenizer, X_dic_vocabulary = dic_seq["X"], dic_seq["tokenizer"], dic_seq["dic_vocabulary"]

# %%
dict(list(X_dic_vocabulary.items())[0:6])

# %%
# Preprocess X_test with the same tokenizer
X_test = text2seq(corpus=df_test["text_clean"], fitted_tokenizer=X_tokenizer, maxlen=X_train.shape[1])

# %%
# Add START and END tokens to the summaries (y)
special_tokens = ("<START>", "<END>")
df_train["summary_clean"] = df_train['summary_clean'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])
df_test["summary_clean"] = df_test['summary_clean'].apply(lambda x: special_tokens[0]+' '+x+' '+special_tokens[1])
df_test["summary_clean"][i]

# %%
# Tokenize the summaries (y_train)
dic_seq = text2seq(corpus=df_train["summary_clean"], top=y_top_words, maxlen=200)

y_train, y_tokenizer, y_dic_vocabulary = dic_seq["X"], dic_seq["tokenizer"], dic_seq["dic_vocabulary"]

# %%
dict(list(y_dic_vocabulary.items())[0:6])

# %%
# Preprocess y_test with the same tokenizer
y_test = text2seq(corpus=df_test["summary_clean"], fitted_tokenizer=y_tokenizer, maxlen=y_train.shape[1])

# %%
# Load pre-trained Word2Vec
nlp = gensim_api.load("glove-wiki-gigaword-300")

# %%
'''
Embeds a vocabulary of unigrams with gensim w2v.
:parameter
    :param dic_vocabulary: dict - {"word":1, "word":2, ...}
    :param nlp: gensim model
:return
    Matric and the nlp model
'''
def vocabulary_embeddings(dic_vocabulary, nlp=None):
    nlp = gensim_api.load("glove-wiki-gigaword-300") if nlp is None else nlp
    embeddings = np.zeros((len(dic_vocabulary)+1, nlp.vector_size))
    for word,idx in dic_vocabulary.items():
        ## update the row with vector
        try:
            embeddings[idx] =  nlp[word]
        ## if word not in model then skip and the row stays all zeros
        except:
            pass
    print("vocabulary mapped to", embeddings.shape[0], "vectors of size", embeddings.shape[1])
    return embeddings

# %%
X_embeddings = vocabulary_embeddings(X_dic_vocabulary, nlp)
X_embeddings.shape

# %%
y_embeddings = vocabulary_embeddings(y_dic_vocabulary, nlp)
y_embeddings.shape

# %%
# Advanced Seq2Seq
lstm_units = 250

##------------ ENCODER (pre-trained embeddings + 3 bi-lstm) ---------------##
x_in = layers.Input(name="x_in", shape=(X_train.shape[1],))
### embedding
layer_x_emb = layers.Embedding(name="x_emb", input_dim=X_embeddings.shape[0], output_dim=X_embeddings.shape[1], 
                               weights=[X_embeddings], trainable=False)
x_emb = layer_x_emb(x_in)
### bi-lstm 1
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4,
                                                  return_sequences=True, return_state=True), 
                                      name="x_lstm_1")
x_out, _, _, _, _ = layer_x_bilstm(x_emb)
### bi-lstm 2
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4,
                                                  return_sequences=True, return_state=True),
                                      name="x_lstm_2")
x_out, _, _, _, _ = layer_x_bilstm(x_out)
### bi-lstm 3 (here final states are collected)
layer_x_bilstm = layers.Bidirectional(layers.LSTM(units=lstm_units, dropout=0.4, recurrent_dropout=0.4, 
                                                  return_sequences=True, return_state=True),
                                      name="x_lstm_3")
x_out, forward_h, forward_c, backward_h, backward_c = layer_x_bilstm(x_out)
state_h = layers.Concatenate()([forward_h, backward_h])
state_c = layers.Concatenate()([forward_c, backward_c])

##------------ DECODER (pre-trained embeddings + lstm + dense) ------------##
y_in = layers.Input(name="y_in", shape=(None,))
### embedding
layer_y_emb = layers.Embedding(name="y_emb", input_dim=y_embeddings.shape[0], output_dim=y_embeddings.shape[1], 
                               weights=[y_embeddings], trainable=False)
y_emb = layer_y_emb(y_in)
### lstm
layer_y_lstm = layers.LSTM(name="y_lstm", units=lstm_units*2, dropout=0.2, recurrent_dropout=0.2,
                           return_sequences=True, return_state=True)
y_out, _, _ = layer_y_lstm(y_emb, initial_state=[state_h, state_c])
### final dense layers
layer_dense = layers.TimeDistributed(name="dense", 
                                     layer=layers.Dense(units=len(y_dic_vocabulary), activation='softmax'))
y_out = layer_dense(y_out)

##---------------------------- COMPILE ------------------------------------##
model = models.Model(inputs=[x_in, y_in], outputs=y_out, name="Seq2Seq")
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

model.summary()

# %%
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()
    plt.savefig("../figs/plot_keras_training.png")

# %%
'''
Fits a keras seq2seq model.
:parameter
    :param X_train: array of sequences
    :param y_train: array of sequences
    :param model: model object - model to fit (before fitting)
    :param X_embeddings: array of weights - shape (len_vocabulary x 300)
    :param y_embeddings: array of weights - shape (len_vocabulary x 300)
    :param build_encoder_decoder: logic - if True returns prediction encoder-decoder
:return
    fitted model, encoder + decoder (if model is noy given)
'''
def fit_seq2seq(X_train, y_train, model=None, X_embeddings=None, y_embeddings=None, build_encoder_decoder=True, epochs=100, batch_size=64, verbose=1):    
    ## model
    if model is None:
        ### params
        len_vocabulary_X, embeddings_dim_X = X_embeddings.shape
        len_vocabulary_y, embeddings_dim_y = y_embeddings.shape
        lstm_units = 250
        max_seq_lenght = X_train.shape[1]
        ### encoder (embedding + lstm)
        x_in = layers.Input(name="x_in", shape=(max_seq_lenght,))
        layer_x_emb = layers.Embedding(name="x_emb", input_dim=len_vocabulary_X, output_dim=embeddings_dim_X, 
                                       weights=[X_embeddings], trainable=False)
        x_emb = layer_x_emb(x_in)
        layer_x_lstm = layers.LSTM(name="x_lstm", units=lstm_units, return_sequences=True, return_state=True)
        x_out, state_h, state_c = layer_x_lstm(x_emb)
        ### decoder (embedding + lstm + dense)
        y_in = layers.Input(name="y_in", shape=(None,))
        layer_y_emb = layers.Embedding(name="y_emb", input_dim=len_vocabulary_y, output_dim=embeddings_dim_y, 
                                       weights=[y_embeddings], trainable=False)
        y_emb = layer_y_emb(y_in)
        layer_y_lstm = layers.LSTM(name="y_lstm", units=lstm_units, return_sequences=True, return_state=True)
        y_out, _, _ = layer_y_lstm(y_emb, initial_state=[state_h, state_c])
        layer_dense = layers.TimeDistributed(name="dense", 
                                             layer=layers.Dense(units=len_vocabulary_y, activation='softmax'))
        y_out = layer_dense(y_out)
        ### compile
        model = models.Model(inputs=[x_in, y_in], outputs=y_out, name="Seq2Seq")
        model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        print(model.summary())
        
    ## train
    training = model.fit(x=[X_train, y_train[:,:-1]], 
                         y=y_train.reshape(y_train.shape[0], y_train.shape[1], 1)[:,1:],
                         batch_size=batch_size, epochs=epochs, shuffle=True, verbose=verbose, validation_split=0.3,
                         callbacks=[callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)])
    if epochs > 1:
        utils_plot_keras_training(training)
    
    # Save training model, encoder and decoder
    training.save("../models/seq2seq.h5")
    
    ## build prediction enconder-decoder model
    if build_encoder_decoder is True:
        lstm_units = lstm_units*2 if any("Bidirectional" in str(layer) for layer in model.layers) else lstm_units
        ### encoder
        encoder_model = models.Model(inputs=x_in, outputs=[x_out, state_h, state_c], name="Prediction_Encoder")
        ### decoder
        encoder_out = layers.Input(shape=(max_seq_lenght, lstm_units))
        state_h, state_c = layers.Input(shape=(lstm_units,)), layers.Input(shape=(lstm_units,))
        y_emb2 = layer_y_emb(y_in)
        y_out2, new_state_h, new_state_c = layer_y_lstm(y_emb2, initial_state=[state_h, state_c])
        predicted_prob = layer_dense(y_out2) 
        decoder_model = models.Model(inputs=[y_in, encoder_out, state_h, state_c], 
                                     outputs=[predicted_prob, new_state_h, new_state_c], 
                                     name="Prediction_Decoder")
        return training.model, encoder_model, decoder_model
    else:
        return training.model
    
    

# %%
# This takes a while
model = fit_seq2seq(X_train, y_train, model, build_encoder_decoder=False, 
                    epochs=100, batch_size=64, verbose=1)

# %%



