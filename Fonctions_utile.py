import re
import nltk
from nltk.corpus import stopwords
import gensim
import spacy


nlp = spacy.load("fr_core_news_md")

"""   2 : retirer caractére spéciaux   """

def sup_caractere_spéciaux(contes):
    contes = contes.lower()
    return  re.sub(r"[,.\"\!@%£#&^*(){}?/;`~:<>+=-\\]","",contes)

def sup_caractere_spéciauxDoc(contes):
    return[sup_caractere_spéciaux(sent)  for sent in contes]

"""   3 : Tokenisation  """
def tokenizeDoc(sentences) :
    for mot in sentences:
        yield(nltk.word_tokenize(str(mot)))
            
def tokenize(sentences) :
    return nltk.word_tokenize(str(sentences))

"""  4 : creation de bigram et trigram français  """
""" seuil (threshold)  plus élevé ==> moins de phrases.
avec 10 on se retourve avec resultat faut (ex: la_maison ) 
avec 100 on a toujours une faute (ex: tu_es)
donc on laisse le threshold a 150 """

def get_bigrams(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    return bigram_model[tokens]


def get_trigrams(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    trigram =  gensim.models.Phrases(bigram[tokens], threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    return trigram_model[bigram_model[tokens]]


def get_bigramsDoc(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    for conte in tokens :
        yield(bigram_model[conte])

def get_trigramsDoc(tokens):
    bigram =  gensim.models.Phrases(tokens, min_count=5, threshold=150)
    trigram =  gensim.models.Phrases(bigram[tokens], threshold=150)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    for conte in tokens :
        yield(trigram_model[bigram_model[conte]])

"""  5 : supprimer les mot d'arret """
mot_arret = stopwords.words('french')

mot_arret.extend([ "à", "afin",'faire',
                   "ah", "ai", "aie", "aient", "aies", "ait", "alors", 'bon', 'peu' ,"veut",
                   "après", "as", "attendu", "au", "delà", "devant", 'fois', 'encore','plus',
                   "aucun", "aucune", "audit", "auprès", "auquel", "aura", "vouloir",
                   "aurai", "auraient", "aurais", "aurait", "auras", "aurez", "met", 'mettre',
                   "auriez", "aurions", "aurons", "auront", "aussi", "autour", "tre",
                   "autre", "autres", "autrui", "aux", "auxdites", "auxdits","trop",
                   "auxquelles", "auxquels", "avaient", "avais", "avait", "avant",
                   "avec", "avez", "aviez", "avions", "avons", "ayant", "ayez", "ayons",
                   "b", "bah", "banco", "ben", "bien", "bé", "c", "c'", "c'est", "c'était",
                   "l'","d'", "etant", "tel", "entre", "si", "ni", "etre", 'ete', 'apres' ,
                   'cela', 'ci','tous', 'trois', 'meme', 'etait','sais' , 'donc' , 'comme' ,
                   'deja', 'assez' , 'depuis' , 'eh' , 'aupres', 'oui', 'non', 'cette' , 'oh' ,'tout' ])

def filtre_motArret(contes) :
    return [token for token  in contes if token not in mot_arret ]

def filtre_motArretDoc(contes) :
    for conte in contes :
        yield([token for token  in conte if token not in mot_arret ])



"""   6 : lemmatisation   """

def lemmatiserDoc(contes, permi=['ADV', 'ADJ','NOUN', 'VERB'] ) :
    for conte in contes :
        yield([token.lemma_ for token in nlp(" ".join(conte)) if token.pos_ in permi])

def lemmatiser(contes, permi=['ADV', 'ADJ','NOUN', 'VERB'] ) :
    return [token.lemma_ for token in nlp(" ".join(contes)) if token.pos_ in permi]


def join(contes):
    text = []
    for conte in contes :
        text.append(' '.join(conte))
    return text




""" Preparer texte """

def preparer(texte) :
    texte = sup_caractere_spéciaux(texte)
    texte = list (tokenize(texte))
    texte = list(get_trigrams(texte))
    texte = list(filtre_motArret(texte))
    texte = list(lemmatiser(texte))
    texte = ' '.join(texte)
    return texte;


