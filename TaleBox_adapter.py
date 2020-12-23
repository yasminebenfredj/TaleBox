from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
import Prediction 



class TaleLogicAdapter(LogicAdapter) :
    '''

    Cette classe represente la logique du TaleBox lorsque l'utilisateur
    veut connaitre le genre de l'histoire.
    
    '''

    def __init__(self, chatbot, **kwargs):
     super().__init__(chatbot, **kwargs)

    def can_process(self, statement):
        mots_fin = ['fini', 'fin', 'terminer', 'sortir', 'quitter', 'théme', 'genre']
        for mot in statement.text.split():
            if mot in mots_fin :
                return True

        return False

    def process(self, input_statement, additional_response_selection_parameters):
        txt =  open('histoire.txt', 'r')
        histoire = txt.read()
        txt.close()
        
        reponse = Prediction.predire_genre(histoire)

        # reponse
        reponse_statement = Statement(text=reponse)
        reponse_statement.confidence = 1

        return reponse_statement
