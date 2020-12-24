from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
import MarkovGeneration as mrkv


class RaconteLogicAdapter(LogicAdapter) :
    '''

    Cette classe represente la logique du TaleBox lorsque l'utilisateur
    est entrain de raconter l'histoire.
    
    '''

    def __init__(self, chatbot, **kwargs):
     super().__init__(chatbot, **kwargs)

    def can_process(self, statement):

        mots_fin =[
            'bonjour',
            'salut',
            'comment tu vas',
            'trés bien et toi',
            'super, ça va moi aussi',
            'je content que tu vas bien',
            'pas bien',
            'ah je suis desolé',
            'je ne suis pas bien',
            'je suis triste pour toi',
            'qui es tu',
            "je m'appelle TaleBox, je peut crée et raconter une histoire avec toi",
            'tu peut faire quoi',
            'je peut crée et raconter une histoire avec toi',
            "comment commencer l'histoire",
            'je te laisse me donner la prémiere phrase',
            "je souhaite commencer l'histoire",
            'je te laisse commencer alors par une phrase simple',
            'quel genre de phrase',
            "tu n'as qu'à me donner un debut avec le lieu ou le contexte",
            'je sais pas quoi dire',
            'donne moi une phrase avec un lieu ou un contexte quelconque']
        non = [m.split() for m in mots_fin]
        for mots in non:
            if all(mot in statement.text.split() for mot in mots) :
                return False
        return True

    def process(self, input_statement, additional_response_selection_parameters):
        debut , txt =  mrkv.nb_dernier_mot(input_statement.text)
        reponse = mrkv.markov_genere(15, debut, "",input_statement.text )
        # reponse
        reponse_statement = Statement(text=reponse)
        reponse_statement.confidence = 1

        histoire = open('histoire.txt','a')
        n = histoire.write(' '+ input_statement.text+ ' ')
        n = histoire.write(' '+ reponse+ ' ')
        histoire.close()

        return reponse_statement




    
