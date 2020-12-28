import pickle
import spacy
from pathlib import Path
import numpy as np
import pandas as pd
#GENSIM
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import gensim.corpora as corpora

import Fonctions_utile as fct

nlp = spacy.load("fr_core_news_md")

""" Load files """
corpus = None
lda = None
best_lda = None
contes_vectorizer = None

with open('Models\Corpus', 'rb') as file1:
    corpus = pickle.load(file1)

with open('Models\ModelLDA', 'rb') as file2:
    lda =  pickle.load(file2)

with open('Models\Vectorizer', 'rb') as file3:
    contes_vectorizer = pickle.load(file3)

with open('Models\BestModelLDA', 'rb') as file4:
    best_lda = pickle.load(file4)
    
dictionary = corpora.Dictionary.load("Models\Dictionary")


"""  Assignement des genre  """
""" probabilité d'apparition des mot dans chaque genre """
motClef = []
def get_Genre(model , vectorisation , nb_mot=20) :
    tokens = vectorisation.get_feature_names()
    for ind, genre in enumerate(model.components_):
        motClef.append([tokens[i] for i in genre.argsort()[:- nb_mot - 1:-1]])
        
get_Genre(best_lda, contes_vectorizer )

data_genres = pd.DataFrame(motClef)
genres = ['Amour, Royaume',
         'Courage, Fantaisie, Triller',
         'Royaume, Courage, Amour',
         'Drame, Triller, Courage',
         'Fantaisie, Royaume',
         'Royaume, Musique',
         'Vie, Sagesse, Aventure',
         'Enfant, Famille',
         'Nature, Fantaisie',
         'Aventure, Animeaux']

data_genres.index = genres
data_genres.index.name = 'Genre'

""" Enregistrement """
'''
data_genres.to_csv("Données\TaleBox_genres.csv", encoding ='ISO-8859-1')
'''

"""  Prédiction des genres """
def predire_genre(texte, nlp = nlp ) :
    '''préparer le texte'''
    texte = fct.preparer_donnée_(texte)
    texte = ' '.join(texte)
    texte = contes_vectorizer.transform([texte])

    ''' LDA Transform '''
    genre_probability = best_lda.transform(texte)
    ind = np.argmax(genre_probability)
    return genres[ind]

""""""""""""""""""

def predire_genre2(texte):
    '''préparer le texte'''
    texte = fct.preparer_donnée_(texte)
    txt_vec = []
    txt_vec = dictionary.doc2bow(texte) #construire sac de mot pour avoir occurence des mots
    genre_vec = []
    genre_vec = lda[txt_vec]

    word_count = np.empty((len(genre_vec), 2), dtype = np.object)
   
    for i in range(len(genre_vec)):
        word_count[i, 0] = genre_vec[i][0]
        word_count[i, 1] = genre_vec[i][1]

    idx = np.argsort(word_count[:, 1]) #trie %
    idx = idx[::-1]
    word_count = word_count[idx] #mot correspondant à % + élévé
    genres = []
    genres = lda.print_topic(word_count[0, 0], 10)
    genres = genres.split('"')
    return ' , '.join([genres[1],genres[3],
                       genres[5],genres[7],
                       genres[9],genres[11],
                       genres[13],genres[15],
                       genres[17],genres[19]])

""" Enregistrement """
'''
data_contes = pd.read_csv(r"Données\TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
data_contes['genre'] = data_contes['conte'].apply(lambda x: predire_genre(x))
data_contes.to_csv("Données\TaleBox_contes.csv", encoding ='ISO-8859-1')

'''
