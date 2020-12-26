from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
import  MarkovGeneration as mkv
import pickle

class QuitteLogicAdapter(LogicAdapter) :
    '''

    Cette classe represente la logique du TaleBox lorsque l'utilisateur
    veut connaitre le genre de l'histoire.
    
    '''

    def __init__(self, chatbot, **kwargs):
     super().__init__(chatbot, **kwargs)

    def can_process(self, statement):
        mots_fin = [['terminer',"l'histoire"],
                    ['fini', "l'histoire"],
                    ['veux', 'genre' ],
                    ['voudrais', 'terminer'],
                    ['genre', "l'histoire"],
                    ['sortir', 'jeu'],
                    ['sortir', 'veut'],
                    ['veut', 'quitter'],
                    ['quit'],
                    ['arrêter', "l'histoire"]]
        
        """  Enregister mots_quit """
        with open('mots_quit', 'wb') as file:
            pickle.dump(mots_fin, file)
            
        for mots in mots_fin:
            if all(mot in statement.text.split() for mot in mots) :
                return True

        return False

    def process(self, input_statement, additional_response_selection_parameters):
        txt =  open('histoire.txt', 'r')
        histoire = txt.read()
        txt.close()
        reponse = "\nHistoire complète : \n"+histoire+".\n\n"
        reponse += "Cette histoire me semble tourner autour des thémes "
        reponse += Prediction.predire_genre(histoire)
        # reponse
        reponse_statement = Statement(text=reponse+".")
        reponse_statement.confidence = 1

        return reponse_statement
