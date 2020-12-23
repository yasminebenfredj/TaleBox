from gensim.corpora import Dictionary
import pickle
from pathlib import Path
from gensim.models import LdaModel
import gensim.corpora as corpora
import Fonctions_utile as fct
import numpy as np
import spacy
import pandas as pd

nlp = spacy.load("fr_core_news_md")

""" Load files """
corpus = None
lda = None
best_lda = None
contes_vectorizer = None

with open('corpus', 'rb') as file1:
    corpus = pickle.load(file1)

with open('modelLDA', 'rb') as file2:
    lda =  pickle.load(file2)

with open('Vectorizer', 'rb') as file3:
    contes_vectorizer = pickle.load(file3)

with open('BestLDA', 'rb') as file4:
    best_lda = pickle.load(file4)
    
dictionary = corpora.Dictionary.load("dictionary")


"""   8.2 : Assignement des genre  """
""" probabilité d'apparition des mot dans chaque genre """
motClef = []
def get_Genre(model , vectorisation , nb_mot=20) :
    tokens = vectorisation.get_feature_names()
    for ind, genre in enumerate(model.components_):
        motClef.append([tokens[i] for i in genre.argsort()[:- nb_mot - 1:-1]])
        
get_Genre(best_lda, contes_vectorizer )

data_genres = pd.DataFrame(motClef)
genres = ['Famille, Courage',
         'Vie, Aventure, Triller',
         'Fantastique, Royaume, Aventure',
         'Fantastique, Merveilleu, Royaume',
         'Drame, Royaume, Merveilleu' ,
         'Courage, Aventure, Decouverte',
         'Merveilleu, Filles, Heureux',
         'Amour, Fantastique, Aventure',
         'Famille, Vie, Animeau',
         'Animeaux, Aventure']

data_genres.index = genres
data_genres.index.name = 'Genre'

"""  Prédiction des genres """

def predire_genre(texte, nlp = nlp ) :
    '''préparer le texte'''

    texte = fct.preparer(texte)
    texte = contes_vectorizer.transform([texte])

    ''' LDA Transform '''
    genre_probability = best_lda.transform(texte)
    ind = np.argmax(genre_probability)
    return genres[ind]

""""""""""""""""""

def predire_genre2(texte):
    texte = fct.sup_caractere_spéciaux(texte)
    texte = list (fct.tokenize(texte))
    texte = list(fct.get_trigrams(texte))
    texte = list(fct.filtre_motArret(texte))
    texte = list(fct.lemmatiser(texte))
    important_words = []
    important_words = texte
    txt_vec = []
    txt_vec = dictionary.doc2bow(important_words) #construire sac de mot pour avoir occurence des mots
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
    genres = lda.print_topic(word_count[0, 0], 3)
    genres = genres.split('"')
    return ' , '.join([genres[1],genres[3], genres[5]])

""" Enregistrement """
'''
data_contes = pd.read_csv(r"TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
data_contes['genre'] = data_contes['conte'].apply(lambda x: predire_genre(x))
data_contes.to_csv("TaleBox_contes.csv", encoding ='ISO-8859-1')

data_genres.to_csv("TaleBox_genres.csv", encoding ='ISO-8859-1')
'''
