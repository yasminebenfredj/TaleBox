import tkinter
from tkinter import *
import TaleBox
import Fonctions_utile as fct

class interface : 
    tale_box  =  Tk()
    box = Text(tale_box)
    reponse_entre = Text(tale_box)
    bouton = Button(tale_box)
    #colors
    Dark = '#374255'
    Light = '#F0F3F4'
    Beige = '#D2C0A7'
    Ivory = '#EEEDE7'
    Tan = '#B68D40'
    
    def __init__(self):
        self.tale_box.title("TaleBox")

    def setLogo(self):
        self.tale_box.iconbitmap("TaileBox_Icon.ico")

    def setColor(self,):
        self.tale_box.config(bg = self.Dark) 

    def entree(self):
        #Bouton "Répondre
        bouton = Button(self.tale_box, font=("Verdana",12,'bold'),text = "Répondre", width="10", height=2 ,
                 bd=0, bg=self.Beige,command= self.envoie )
        bouton.place(x=575, y=445, height=40)

        #Entree de repense 
        self.reponse_entre = Text(self.tale_box, font=("Helvetica",13, 'normal','italic'))
        self.reponse_entre.place(x=20, y = 445 , height = 40, width = 500)
        self.reponse_entre.bind("<Return>", self.envoie)

    def tale_avance(self):
        self.box.config(bd=0,bg = self.Ivory, height="8", width="50", font="Arial")
        self.box.place(x = 20, y = 20, height = 400, width = 670)
        
        self.box.tag_add("tag1", "1.5", "1.9")
        self.box.insert(END, "Bot : \n" ,"tag1")
        self.box.insert(END, "Bonjours, entrez une histoire et je vous donne le genre.  \n\n" )
        self.box.tag_config("tag1",font=("Verdana",14, 'bold'), foreground= self.Dark)
        self.box.config(foreground= self.Dark, font=("Verdana", 10 ))
        self.box.config(state=DISABLED)
                
    def envoie(self) :
        #Insertion de message et repense 
        msg = self.reponse_entre.get("1.0",'end-1c').strip()
        self.reponse_entre.delete("0.0",END)
        self.box.tag_add("tag1", "1.5", "1.9")
        self.box.config(state=NORMAL)

        if msg != '':
            self.box.config(state=NORMAL)
            self.box.insert(END, "Vous : \n" ,"tag1")
            self.box.insert(END, ""+ msg + '\n\n')
            pred = TaleBox.repondre(msg)
            
            #reponce = getReponse(pred, intents)

            self.box.insert(END, "Bot : \n" ,"tag1")
            self.box.insert(END, ""+ pred.text + '\n\n' )
            self.box.config(state=DISABLED)
            
            msg =  ''
            pred = ''
            self.box.yview(END)
 
    def affiche(self):
        self.tale_box.geometry("715x500")
        self.tale_box.maxsize(800,600)
        self.tale_box.minsize(300,400)

        self.setLogo()
        self.setColor()
        self.entree()
        self.tale_avance()

        self.envoie()
        self.tale_box.mainloop()

class Repondre_Prediction:
    message =''
    
    def __init__(self,message):
        self.message = message

def main():
    taleBox = interface()
    taleBox.affiche()

if __name__ == "__main__":
    histoire = open('histoire.txt','w')
    histoire.close()
    main()
