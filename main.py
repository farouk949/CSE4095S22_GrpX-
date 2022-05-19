import numpy as np
from scipy._lib._tmpdirs import in_dir
from sklearn import svm, discriminant_analysis, dummy
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor, _gb_losses
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVR
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import string
import glob
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import TrigramAssocMeasures
from nltk import word_tokenize
from nltk import FreqDist
from collections import defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,Bidirectional
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.layers import Dropout
from nltk.corpus import stopwords
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import re
import os
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import gensim
import gensim.downloader as api

for dirname, _, filenames in os.walk('/dataset2021-01/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return(data)

def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dumb(data, f, indent=4)


def remove_stops(text, stops):
    text = re.sub(r"AC\/\d{1,4}\/\d{1,4}","",text)
    words = text.split()
    final = []
    for word in words:
        if word not in stops:
            final.append(word)
    final = " ".join(final)
    final = final.translate(str.maketrans("", "", string.punctuation))
    final = "".join([i for i in final if not i.isdigit()])
    while "  " in final:
        final = final.replace("  ", " ")

def clean_docs(docs):
    stops = stopwords.words("english") + stopwords.words('turkish')
    months = load_data("months")

    stops = stops + months
    final = []
    for doc in docs:
        clean_doc = remove_stops(doc, stops)
        final.append(clean_doc)
        return(final)

final_stopwords_list = stopwords.words('english') + stopwords.words('turkish')

descriptions = load_data("dataset2021-01/1.json")["ictihat"]
cleaned_docs = clean_docs(descriptions)
print (descriptions)

sns.set()
sns.countplot(descriptions)
print(plt.show())

EMBEDDING_DIM = 100 # this means the embedding layer will create  a vector in 100 dimension
model = Sequential()
model.add(Embedding(input_dim = num_words,# the whole vocabulary size
                          output_dim = EMBEDDING_DIM, # vector space dimension
                          input_length= X_train_pad.shape[1] # max_len of text sequence
                          ))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(200,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(CuDNNLSTM(100,return_sequences=False)))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = 'accuracy')

es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

history_embedding = model.fit(X_train_pad,y_train, epochs = 35, batch_size = 120, validation_data=(X_test_pad, y_test),verbose = 1, callbacks= [es, mc]  )

plt.plot(history_embedding.history['accuracy'],c='b',label='train accuracy')
plt.plot(history_embedding.history['val_accuracy'],c='r',label='validation accuracy')
plt.legend(loc='lower right')
plt.show()

glove_gensim  = api.load('glove-wiki-gigaword-100') # this would download vector with 100 dimension

vector_size = 100
gensim_weight_matrix = np.zeros((num_words ,vector_size))
gensim_weight_matrix.shape

for word, index in tokenizer.word_index.items():
    if index < num_words: # since index starts with zero
        if word in glove_gensim.wv.vocab:
            gensim_weight_matrix[index] = glove_gensim[word]
        else:
            gensim_weight_matrix[index] = np.zeros(100)

model_gensim.summary()

#print (cleaned_docs)

vectorizer = TfidfVectorizer(lowercase=True, max_features=100, max_df=0.8, min_df=5,
                             ngram_range=(1,3), stop_words=final_stopwords_list)

vectors = vectorizer.fit_transform(descriptions)
feature_names = vectorizer.get_feature_names_out()
#dense = vectors.todense()
#denselist = dense.tolist()

#all_keywords = []
#for description in denselist:
    #x=0
    #keywords = []
    #for word in description:
        #if word > 0:
            #keywords.append(feature_names[x])
            #x=x+1
            #all_keywords.append(keywords)

#print(descriptions[0])
#print(all_keywords[0])

#true_k = 20

#model = KMeans(n_clusters=true_k, init="k-means++", max_iter=100, n_init=1)

#model.fit(vectors)

#order_centroids = model.cluster_centers_.argsort()[:, ::-1]
#terms = vectorizer.get_feature_names_out()

with open ("testresults.txt", "w", encoding="utf-8") as f:
    for i in range(true_k):
        f.write(f"Cluster {i}")
        f.write("\n")
        for ind in order_centroids[i, :10]:
            f.write (' %s' % terms[ind],)
            f.write("\n")
        f.write("\n")
        f.write("\n")