import numpy as np
from pandas import DataFrame
import random
import re
import nltk
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import gensim.models 
from nltk.corpus import stopwords

import spacy
nlp = spacy.load("fr_core_news_md")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()



#hardCoded repense :
#from nltk.chat.util import Chat, reflections
##hardCode = [
##    ['(Bonjour | Coucou | Salut )' , ['Salut toi ! ']],
##    ["Comment tu t'appelles ?" , ["Je m'appelle TaleBox"]],
##    ["(Qui est tu ?|Tu sert à quoi ?|Tu sais faire quoi?)", ["Je suis conçu pour crée des contes avec ton aide.", 'Je suis un bot qui crée des contes.' , "Je m'appelle TaleBox et je suis la pour crée un conte avec toi." ]],
##    ["(Commençant|commencer|Je veux commencer|Comment Commencer?)", ["Tu veux qu'on commence notre histoire?", "Pour commencer, tu peut me donner un lieu ?"]]] 
##
##
##chat = Chat(hardCode , reflections)
##chat.converse()




class Repondre_Prediction:
    message =''
    
    def __init__(self,message):
        self.message = message





############### 1 : Import Dataset

data_contes = pd.read_csv(r"C:\Users\Jasmine\Desktop\Master 1\Semestre 1\TATIA\Projet\TaleBox\TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
data_genres = pd.read_csv(r"C:\Users\Jasmine\Desktop\Master 1\Semestre 1\TATIA\Projet\TaleBox\TaleBox_genres.csv", sep=',', encoding ='ISO-8859-1')

contes = data_contes['conte'].tolist()
genres = data_contes['genre'].tolist()
############### 2 : retirer caractére spéciaux 

# Remove new line characters
contes= [re.sub('\s+', ' ', sent) for sent in contes]


############### 3 : Tokenisation 

def tokenize(sentences) :
    for mot in sentences:
        yield(simple_preprocess(str(mot), deacc=True))

tokens = list(tokenize(contes))

############## 4 : creation de bigram et trigram français

bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
trigram =  gensim.models.Phrases(bigram[tokens], threshold=150)  

# seuil ( threshold)  plus élevé ==> moins de phrases.
#avec 10 on se retourve avec resultat faut (ex: la_maison ) ,
#avec 100 on a toujours une faute comme "tu_es"
#donc on laisse le threshold a 150


bigram_model = gensim.models.phrases.Phraser(bigram)
trigram_model = gensim.models.phrases.Phraser(trigram)

def get_bigrams(data):
    for conte in data :
        yield(bigram_model[conte])

def get_trigrams(data):
    for conte in data :
        yield(trigram_model[bigram_model[conte]])

contes_bigrams = list(get_bigrams(tokens))
contes_trigrams = list(get_trigrams(tokens))


############### 5 : supprimer les mot d'arret

mot_arret = stopwords.words('french')
mot_arret.extend([ "a", "afin",
                   "ah", "ai", "aie", "aient", "aies", "ait", "alors",
                   "après", "as", "attendu", "au", "delà", "devant",
                   "aucun", "aucune", "audit", "auprès", "auquel", "aura",
                   "aurai", "auraient", "aurais", "aurait", "auras", "aurez",
                   "auriez", "aurions", "aurons", "auront", "aussi", "autour",
                   "autre", "autres", "autrui", "aux", "auxdites", "auxdits",
                   "auxquelles", "auxquels", "avaient", "avais", "avait", "avant",
                   "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                   "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était",
                   "l'","d'", "etant", "tel", "entre", "si", "ni", "etre", 'ete', 'apres' ,
                   'cela', 'ci','tous', 'trois', 'meme', 'etait','sais' , 'donc' , 'comme' ,
                   'deja', 'assez' , 'depuis' , 'eh' , 'aupres', 'oui', 'non', 'cette' , 'oh'  ])

def filtre_motArret(contes) :
    for conte in contes :
        yield([token for token  in simple_preprocess(str(conte)) if token not in mot_arret ])


contes_filtrer = list(filtre_motArret(contes_trigrams))

################ 6 : lemmatisation

def lemmatiser(contes, permi=['ADV', 'ADJ','NOUN', 'VERB'] ) :
    for conte in contes :
        yield([token.lemma_ for token in nlp(" ".join(conte)) if token.pos_ in permi])

contes_lemmatiser = list(lemmatiser(contes_filtrer ))

#print(contes_filtrer[5], len(contes_filtrer[5]))
#print(contes_lemmatiser[5], len(contes_lemmatiser[5]))



################7 :  cluster de mots pour la prédiction du sujet

def join(contes):
    text = []
    for conte in contes :
        text.append(' '.join(conte))
    return text
        


from sklearn.feature_extraction.text import CountVectorizer

contes_lemmatiser_df = DataFrame( join(contes_lemmatiser),columns=['Contes'])
contes_vectorizer = CountVectorizer(stop_words = mot_arret, ngram_range=(1, 4), min_df = 5, max_df = 0.8 )
#transformer en matrice de nombre de token 
contes = contes_vectorizer.fit_transform(contes_lemmatiser_df['Contes'])




############## 8 : Recherche des meilleurs hyperparametre pour notre model



from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV


parameters = [{'n_components': [ 5, 6,7,9,10, 15, 20, 25]}]
# le grid search va nous dire lequelle est le meilleur nombre de type de sujet 
              
#model LDA non supervisé pour avoir proba des mots
lda = LDA()
              
# Grid Search 
model = GridSearchCV(lda,parameters)
              
#Fit the model
model.fit(contes)
best_lda = model.best_estimator_



def get_Genre(model , vectorisation , top_mot):
    tokens = vectorisation.get_feature_names()
    for ind, genre in enumerate(model.components_):
        print("\nGenre {}:".format(ind))
        print(" ".join([tokens[i] for i in genre.argsort()[:- top_mot - 1:-1]]))
        



get_Genre(best_lda, contes_vectorizer ,10)

























