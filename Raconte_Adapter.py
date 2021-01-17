from chatterbot.logic import LogicAdapter
from chatterbot.conversation import Statement
import  MarkovGeneration as mkv
import pickle

class RaconteLogicAdapter(LogicAdapter) :
    '''

    Cette classe represente la logique du TaleBox lorsque l'utilisateur
    est entrain de raconter l'histoire.
    
    '''

    def __init__(self, chatbot, **kwargs):
     super().__init__(chatbot, **kwargs)

    def can_process(self, statement):
        mots1 , mots2 = [] , []
        with open('Données\intents', 'rb') as file1:
            mots1 = pickle.load(file1)
        with open('Données\mots_quit', 'rb') as file2:
            mots2 = pickle.load(file2)
            
        non = [m.split() for m in mots1]
        non.extend(mots2)
        
        for mots in non:
            if all(mot in statement.text.split() for mot in mots) :
                return False
        return True

    def process(self, input_statement, additional_response_selection_parameters):
        debut , txt =  mkv.nb_dernier_mot(input_statement.text)
        reponse , ajout_histoire = mkv.markov_genere(12, debut, "" )
        # reponse
        reponse_statement = Statement(text=reponse)
        reponse_statement.confidence = 1

        histoire = open('histoire.txt','a' , encoding="utf-8")
        n = histoire.write(' '+ input_statement.text+ ' ')
        if ajout_histoire :
            n = histoire.write(' '+ reponse+ ' ')
        histoire.close()

        return reponse_statement




    
