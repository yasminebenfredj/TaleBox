import random
import re
from pandas import DataFrame 
import pandas as pd
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
import gensim.models 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV
import pickle
from gensim.models import LdaModel
import Fonctions_utile as fct
from gensim.corpora import Dictionary
from gensim.test.utils import datapath
from pathlib import Path
import numpy as np 

nlp = spacy.load("fr_core_news_md")


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

lda = LdaModel(corpus=corpus,id2word=dictionary,
               num_topics=num_topics)


"""  8.1 : Enregister """
with open('corpus', 'wb') as file1:
    pickle.dump(corpus, file1)

dictionary.save("dictionary")

with open('modelLDA', 'wb') as file2:
    pickle.dump(lda,file2)



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

""" Ici je vais faire une autre maniére de prédire le genre d'une histoire """

contes_lemmatiser_df = DataFrame( fct.join(contes_lemmatiser),columns=['Contes'])
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
genre_num = []
motClef = []
def get_Genre(model , vectorisation , nb_mot):
    tokens = vectorisation.get_feature_names()
    for ind, genre in enumerate(model.components_):
        genre_num.append("Genre {} : ".format(ind))
        motClef.append([tokens[i] for i in genre.argsort()[:- nb_mot - 1:-1]])
        
get_Genre(best_lda, contes_vectorizer ,20)

"""  8.2 : predire le sujet d'un nouveau texte   """
""" probabilité d'apparition des mot dans chaque genre """

genre_motClef = pd.DataFrame(motClef)
genres = ['Famille , Enfant',
         'Vie , Aventure',
         'Fantastique , Royaume',
         'Aventure , Animeaux',
         'Royaume , Nature , Merveilleu' ,
         'Amour , Drame',
         'Drame , Famille',
         'Aventure , Nature , Solitude',
         'Courage , Fantastique',
         'Aventure , Famille']


genre_motClef.index = genres
genre_motClef.to_csv("TaleBox_genres.csv")

"""   9.2 : Assignement des genres """

def predire_genre(texte, nlp = nlp ) :
    '''préparer le texte'''
    texte= text1
    texte = fct.sup_caractere_spéciaux(texte)
    texte = list (fct.tokenize(texte))
    texte = list(fct.get_trigrams(texte))
    texte = list(fct.filtre_motArret(texte))
    texte = list(fct.lemmatiser(texte))
    texte = ' '.join(texte)
    texte = contes_vectorizer.transform([texte])

    ''' LDA Transform '''
    genre_probability = best_lda.transform(texte)
    ind = np.argmax(genre_probability)
    return genres[ind]


""" 10.2 :  testes """
text1 = contes_lemmatiser_df['Contes'][2]

text2 = contes_lemmatiser_df['Contes'][4]
""" """

#predire_genre(texte=text1)
#predire_genre(texte=text2)

