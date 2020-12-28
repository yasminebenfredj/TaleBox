import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.response_selection import get_first_response
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.trainers import ChatterBotCorpusTrainer
import pickle

# BestMatch:  cette methode retourne la meilleur répense (chatbot,liste mot exclus)

Tale_Bot = ChatBot(name='TaleBot', read_only=False, logic_adapters=[
    {"import_path": "Raconte_Adapter.RaconteLogicAdapter"},
    {"import_path": "Quitte_Adapter.QuitteLogicAdapter"},
    
    {'import_path': 'chatterbot.logic.BestMatch',
    'default_response': "Désolé, je n'ai pas compris. Veuillez me donner plus d'informations.",
    'maximum_similarity_threshold': 0.80}
    ])
        
intents = ['Bonjour',
           'Salut, comment tu vas ?',
           'Super bien et toi comment tu vas ?',
           'Trés bien',
           'Super, ça va moi aussi',
           'Je content que tu vas bien',
           'Pas bien !',
           'Ah! je suis desolé, je croit que je peut vous aider avec mes histoires..."',
           'Je ne suis pas bien !',
           'Je suis triste pour toi !',
           'Qui es tu ?',
           "Je m'appelle TaleBox, je peut crée et raconter une histoire avec toi",
           'Tu peut faire quoi ?',
           'Je peut crée et raconter une histoire avec toi',
           "Comment pouvez-vous m'aider?",
           'Je peut crée une histoire avec toi',
           "Comment commencer l'histoire",
           'Je te laisse me donner la prémiere phrase',
           "Je souhaite commencer l'histoire",
           'Je te laisse commencer alors par une phrase simple',
           'Quel genre de phrase ?',
           "Tu n'as qu'à me donner un debut avec le lieu ou le contexte",
           'Je sais pas quoi dire',
           'Donne moi une phrase avec un lieu ou un contexte quelconque',
           "Merci c'est gentil",
           "Le plaisir est pour moi",
           "Merci",
           "De rien! Le plaisir est pour moi",
           'Au revoir',
           "À la prochaine!",
           "À bientôt",
           "Au revoir! Revenez bientôt",
           "Bye-Bye! Bonne journée"]

"""  Enregister intents  """
with open('Données\intents', 'wb') as file:
    pickle.dump(intents, file)

trainer = ListTrainer(Tale_Bot)
trainer.train(intents)

#corpus = ChatterBotCorpusTrainer(Tale_Bot)
#corpus.train('chatterbot.corpus.french')

def repondre(msg) :
    rep = Tale_Bot.get_response(msg)
    return rep



