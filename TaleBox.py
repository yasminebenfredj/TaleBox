import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.response_selection import get_first_response
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.trainers import ChatterBotCorpusTrainer


# BestMatch:  cette methode retourne la meilleur répense (chatbot,liste mot exclus)

Tale_Bot = ChatBot(name='TaleBot', read_only=True, logic_adapters=[
    {"import_path": "Quitte_Adapter.QuitteLogicAdapter"},
    {"import_path": "Raconte_Adapter.RaconteLogicAdapter"},
    {'import_path': 'chatterbot.logic.BestMatch',
    'default_response': "Désolé, j'ai pas compris est ce que vous pouvez reformulez.",
    'maximum_similarity_threshold': 0.50}
    ])
        

intents = ['bonjour',
           'salut',
           'comment tu vas ?',
           'trés bien et toi ?',
           'super, ça va moi aussi.',
           'je content que tu vas bien.',
           'pas bien !',
           'ah je suis desolé .',
           'je ne suis pas bien !',
           'je suis triste pour toi !',
           'qui es tu ?',
           "je m'appelle TaleBox, je peut crée et raconter une histoire avec toi .",
           'tu peut faire quoi ?',
           'je peut crée et raconter une histoire avec toi .',
           "comment commencer l'histoire",
           'je te laisse me donner la prémiere phrase ...',
           "je souhaite commencer l'histoire",
           'je te laisse commencer alors par une phrase simple ...',
           'quel genre de phrase ?',
           "tu n'as qu'à me donner un debut avec le lieu ou le contexte .",
           'je sais pas quoi dire .',
           'donne moi une phrase avec un lieu ou un contexte quelconque .']


trainer = ListTrainer(Tale_Bot)
for intent in intents :
    trainer.train(intent)


#corpus = ChatterBotCorpusTrainer(Tale_Bot)
#corpus.train('chatterbot.corpus.french')

def repondre(msg) :
    rep = Tale_Bot.get_response(msg)
    print(rep)
    return rep



