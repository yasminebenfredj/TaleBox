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
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV

nlp = spacy.load("fr_core_news_md")
lemmatizer = WordNetLemmatizer()

"""   1 : Import Dataset   """

data_contes = pd.read_csv(r"TaleBox_contes.csv", sep=',', encoding ='ISO-8859-1')
data_genres = pd.read_csv(r"TaleBox_genres.csv", sep=',', encoding ='ISO-8859-1')
contes = data_contes['conte'].tolist()
genres = data_contes['genre'].tolist()

"""   2 : retirer caractére spéciaux   """

def sup_caractere_spéciaux(contes):
    return [re.sub('\s+', ' ', sent) for sent in contes]
 
contes = sup_caractere_spéciaux(contes)


"""   3 : Tokenisation  """

def tokenize(sentences) :
    for mot in sentences:
        yield(simple_preprocess(str(mot), deacc=True))

tokens = list(tokenize(contes))

"""  4 : creation de bigram et trigram français  """
""" seuil (threshold)  plus élevé ==> moins de phrases.
avec 10 on se retourve avec resultat faut (ex: la_maison ) 
avec 100 on a toujours une faute (ex: tu_es)
donc on laisse le threshold a 150 """

def get_bigrams(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)

    for conte in tokens :
        yield(bigram_model[conte])

def get_trigrams(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    trigram =  gensim.models.Phrases(bigram[tokens], threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)

    for conte in tokens :
        yield(trigram_model[bigram_model[conte]])

contes_bigrams = list(get_bigrams(tokens))
contes_trigrams = list(get_trigrams(tokens))



"""  5 : supprimer les mot d'arret """

mot_arret = stopwords.words('french')
mot_arret.extend([ "a", "afin",
                   "ah", "ai", "aie", "aient", "aies", "ait", "alors", 'bon', 'peu' ,"veut"
                   "après", "as", "attendu", "au", "delà", "devant", 'fois', 'encore',
                   "aucun", "aucune", "audit", "auprès", "auquel", "aura", "vouloir"
                   "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "met", 'mettre'
                   "auriez", "aurions", "aurons", "auront", "aussi", "autour", "tre",
                   "autre", "autres", "autrui", "aux", "auxdites", "auxdits",
                   "auxquelles", "auxquels", "avaient", "avais", "avait", "avant",
                   "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                   "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était",
                   "l'","d'", "etant", "tel", "entre", "si", "ni", "etre", 'ete', 'apres' ,
                   'cela', 'ci','tous', 'trois', 'meme', 'etait','sais' , 'donc' , 'comme' ,
                   'deja', 'assez' , 'depuis' , 'eh' , 'aupres', 'oui', 'non', 'cette' , 'oh' ,'tout' ])

def filtre_motArret(contes) :
    for conte in contes :
        yield([token for token  in simple_preprocess(str(conte)) if token not in mot_arret ])


contes_filtrer = list(filtre_motArret(contes_trigrams))


"""   6 : lemmatisation   """

def lemmatiser(contes, permi=['ADV', 'ADJ','NOUN', 'VERB'] ) :
    for conte in contes :
        yield([token.lemma_ for token in nlp(" ".join(conte)) if token.pos_ in permi])

contes_lemmatiser = list(lemmatiser(contes_filtrer ))



"""   7 :  cluster de mots pour la prédiction du sujet  """

def join(contes):
    text = []
    for conte in contes :
        text.append(' '.join(conte))
    return text
        


contes_lemmatiser_df = DataFrame( join(contes_lemmatiser),columns=['Contes'])
contes_vectorizer = CountVectorizer(stop_words = mot_arret, ngram_range=(1, 4), min_df = 5, max_df = 0.8 )
contes = contes_vectorizer.fit_transform(contes_lemmatiser_df['Contes'])

"""   8 : Recherche des meilleurs hyperparametre pour notre model  """
""" le grid search va nous donnée le meilleur nombre de genre """
         
#model LDA non supervisé pour avoir proba des mots
lda = LDA()
              
# Grid Search
parameters = [{'n_components': [6,7,9,10,15,20,25]}]
model = GridSearchCV(lda,parameters)
              
#Fit the model
model.fit(contes)
best_lda = model.best_estimator_

genre_num = []
motClef = []
def get_Genre(model , vectorisation , top_mot):
    tokens = vectorisation.get_feature_names()
    for ind, genre in enumerate(model.components_):
        genre_num.append("Genre {}:".format(ind))
        motClef.append([tokens[i] for i in genre.argsort()[:- top_mot - 1:-1]])
        #print("\nGenre {}:".format(ind))
        #print(" ".join([tokens[i] for i in genre.argsort()[:- top_mot - 1:-1]]))
        
get_Genre(best_lda, contes_vectorizer ,20)

"""  9 : predire le sujet d'un nouveau texte   """
""" probabilité d'apparition des mot dans chaque genre """

genre_motClef = pd.DataFrame(motClef)
#genre_motClef.columns = contes_vectorizer.get_feature_names()
genre_motClef.index = genre_num

""""""""""""""""""""""""""""""""""""""""""""""""

def predire_genre(texte, nlp = nlp ) :
    '''préparer le texte'''
    texte = sup_caractere_spéciaux(texte)
    texte = list (tokenize(texte))
    texte = list(get_trigrams(texte))
    texte = list(filtre_motArret(texte))
    texte = list(lemmatiser(texte))
    texte = join(texte)
    texte = contes_vectorizer.transform(texte)

    ''' LDA Transform '''
    genre_probability_scores = best_lda.transform(texte)
    genre = genre_motClef.iloc[np.argmax(genre_probability_scores), :].values.tolist()
    return  genre, genre_probability_scores

