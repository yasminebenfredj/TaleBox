import numpy as np
import os
import re
import random
import pandas as pd
import string
import Fonctions_utile as fct


"""   1 : Import Dataset   """
data_contes = pd.read_csv(r"Données\TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
data = data_contes['conte'].tolist()

"""   2 : retirer caractére spéciaux   """
data = fct.sup_caractere_spéciauxDoc(data)

"""   3 : Tokenisation  """
tokens = list(fct.tokenizeDoc(data))

"""  4 : creation de bigram et trigram français  """
contes_bigrams = list(fct.get_bigramsDoc(tokens))
contes_trigrams = list(fct.get_trigramsDoc(contes_bigrams))

"""  Ici nous n'allons pas lemmetiser """

contes = []
for conte in contes_trigrams :
        contes.extend(conte)

#contexte = 2 ou 3
#j'ai choisi un contexte de 3 mots par defaut pour avoir le maximum de sens 
contexte = 3
def markov_model(contes, contexte=contexte):
    model = {}
    for n in range(len(contes) - contexte - 2):
        current , new = "", ""
        for x in range(contexte):
            current += contes[n + x] + " "
            new += contes[n + x + contexte] + " "
        current = current[:-1]
        new = new[:-1]
        if current not in model :
            model[current] = {}
            model[current][new] = 1
        else :
            if new in model[current]:
                model[current][new] += 1
            else:
                model[current][new] = 1
    for current , proba in model.items():
        total = sum(proba.values())
        for new, nb in proba.items():
            model[current][new] = nb/total
    return model

model = markov_model(contes)

def markov_genere(long, debut, texte):
    debut = debut.lower()    
    if long == 0 or len(debut.split()) < contexte:
        texte += "..."
        return texte, True
        
    try:
        index  = list(model[debut].values()).index(max(model[debut].values()))
        new = list(model[debut].keys())[index]
        texte += new + " "
        return markov_genere(long-1, new, texte)
    
    except KeyError:
        return "Vous êtes tellement doué que je n'ai pas d'idée pour la suite, continuez ...", False
        

def nb_dernier_mot(texte, nb=contexte) :
    resp = ""
    nb_esp = 0
    new_texte = texte
    for i in range(1,len(texte)):
        new_texte = new_texte[0:-1]
        if texte[-i] == " ":
            nb_esp += 1
        if nb_esp >= nb:
            break
        resp = texte[-i]+resp
    return resp , new_texte
        
