from gensim.corpora import Dictionary
import pickle
from pathlib import Path
from gensim.models import LdaModel
import gensim.corpora as corpora
import TaleBox
import Fonctions_utile as fct
import numpy as np

""" Load files """
corpus = None
lda = None
with open('corpus', 'rb') as file1:
    corpus = pickle.load(file1)

with open('modelLDA', 'rb') as file2:
    lda =  pickle.load(file2)
    
dictionary = corpora.Dictionary.load("dictionary")

""" Pour testes"""
txt =  open('C:/Users/Jasmine/Desktop/hisoire.txt', 'r')
text1 = txt.read()
txt.close()
txt =  open('C:/Users/Jasmine/Desktop/histoire.txt', 'r')
text2 = txt.read()
txt.close()
""" """


def predire_genre(texte):
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

