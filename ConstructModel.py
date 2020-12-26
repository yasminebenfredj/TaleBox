""" Dont run this file """

import re
import pickle
import pandas as pd
from pathlib import Path
import numpy as np 
#GENSIM
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
import gensim.models 
from gensim.models import LdaModel
from gensim.test.utils import datapath
#SKLEARN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV

import Fonctions_utile as fct


"""   1 : Import Dataset   """
data_contes = pd.read_csv(r"Données\TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
contes = data_contes['conte'].tolist()

"""   2 : Retirer caractére spéciaux   """
contes = fct.sup_caractere_spéciauxDoc(contes)

"""   3 : Tokenisation  """
tokens = list(fct.tokenizeDoc(contes))

"""  4 : Création de bigram et trigram français  """
contes_bigrams = list(fct.get_bigramsDoc(tokens))
contes_trigrams = list(fct.get_trigramsDoc(contes_bigrams))

"""  5 : Lemmatisation   """
contes_lemmatiser = list(fct.lemmatiserDoc(contes_trigrams ))

"""  6 : Supprimer les mot d'arret  """
contes_filtrer = list(fct.filtre_motArretDoc(contes_lemmatiser))
###############################################################################

"""  7.1 : Création LDA model  """
dictionary = Dictionary(contes_lemmatiser)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5) #comme faire/dire/puis...
corpus = [dictionary.doc2bow(doc) for doc in contes_lemmatiser]

# Set training parameters.
num_topics = 20

# Make a index to word dictionary.
temp = dictionary[1]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

lda = LdaModel(corpus=corpus,
               id2word=dictionary,
               num_topics=num_topics)


"""  8.1 : Enregister """
with open('Models\Corpus', 'wb') as file1:
    pickle.dump(corpus, file1)

dictionary.save("Models\Dictionary")

with open('Models\ModelLDA', 'wb') as file2:
    pickle.dump(lda,file2)

###############################################################################

""" Ici je vais faire une autre méthode de prédire le genre d'une histoire """
mot_arret = fct.mot_arret

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

"""  8.2 : Enregister """

with open('Models\Vectorizer', 'wb') as file3:
    pickle.dump(contes_vectorizer,file3)

with open('Models\BestModelLDA', 'wb') as file4:
    pickle.dump(best_lda,file4)

