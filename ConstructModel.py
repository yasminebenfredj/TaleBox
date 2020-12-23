""" Dont run this file """

import random
import re
import pandas as pd
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
import gensim.models 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
import pickle
from gensim.models import LdaModel
import Fonctions_utile as fct
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from pathlib import Path
import numpy as np 


"""   1 : Import Dataset   """
data_contes = pd.read_csv(r"TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
contes = data_contes['conte'].tolist()

"""   2 : retirer caractére spéciaux   """
contes = fct.sup_caractere_spéciauxDoc(contes)

"""   3 : Tokenisation  """
tokens = list(fct.tokenizeDoc(contes))

"""  4 : creation de bigram et trigram français  """
contes_bigrams = list(fct.get_bigramsDoc(tokens))
contes_trigrams = list(fct.get_trigramsDoc(tokens))


"""  5 : supprimer les mot d'arret  """
mot_arret = fct.mot_arret
contes_filtrer = list(fct.filtre_motArretDoc(contes_trigrams))


"""   6 : lemmatisation   """
contes_lemmatiser = list(fct.lemmatiserDoc(contes_filtrer ))


"""  7.1 : LDA model  """
dictionary = Dictionary(contes_lemmatiser)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5) #comme faire/dire/puis...
corpus = [dictionary.doc2bow(doc) for doc in contes_lemmatiser]

# Set training parameters.
num_topics = 20
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[1]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

lda = LdaModel(corpus=corpus,
               id2word=dictionary,
               num_topics=num_topics)


"""  8.1 : Enregister """
with open('corpus', 'wb') as file1:
    pickle.dump(corpus, file1)

dictionary.save("dictionary")

with open('modelLDA', 'wb') as file2:
    pickle.dump(lda,file2)

###############################################################################

""" Ici je vais faire une autre méthode de prédire le genre d'une histoire """

contes_lemmatiser_df = pd.DataFrame( fct.join(contes_lemmatiser),columns=['Contes'])
contes_vectorizer = CountVectorizer(stop_words = mot_arret, ngram_range=(1, 4), min_df = 5, max_df = 0.8 )
contes = contes_vectorizer.fit_transform(contes_lemmatiser_df['Contes'])

"""   7.2 : Recherche des meilleurs hyperparametre pour notre model  """
""" le grid search va nous donnée le meilleur nombre de genre """
         
#model LDA non supervisé pour avoir proba des mots
lda = LDA()
              
# Grid Search
parameters = [{'n_components': [10,15,20,25]}]
model = GridSearchCV(lda,parameters)
              
#Fit the model
model.fit(contes)
best_lda = model.best_estimator_
best_lda = best_lda.fit(contes)

"""  9.1 : Enregister """

with open('Vectorizer', 'wb') as file3:
    pickle.dump(contes_vectorizer,file3)

with open('BestLDA', 'wb') as file4:
    pickle.dump(best_lda,file4)

