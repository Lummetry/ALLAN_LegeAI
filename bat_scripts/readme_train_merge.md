# Proces antrenare get_merge

## Parametri de antrenare
Scriptul de antrenare se regaseste ori pe Desktop, ori in folderul “bat_scripts” din repository si are numele “train_merge.bat”. In cadrul rularii acestui script, **utilizatorul trebuie sa ia urmatoarele decizii**: 

1. Path train data folder (default = `C:\Users\damian\Desktop\TRAIN_GET_MERGE_DATA`)
    - path catre folder-ul unde se afla fisierele de training
    - fisierele de training au format .jsonl si sunt obtinute prin Export din interfata de adnotare
    - sunt luate in considerare toate fisierele .jsonl din folder
    
2. Path NER folder (default = `C:\allan_data\MergeNER`)
    - path catre folder-ul unde va fi pus noul model
    - folder-ul poate contine maxim 7 modele; in caz de prea multe modele este sters cel mai vechi
    - in general nu va fi necesar sa fie configurat acest parametru. **Se recomanda sa fie lasat la valoarea default**    

## Proces adnotare (Doccano)
1. Pornire
	* Lansare Docker
	* Selectare Containers > selectare doccano/doccano > start
	* in browser > localhost:8000
	* Log in cu:
		User: admin
		Pass: admin
		
2. Proiect nou
	* Create > selectare Sequence Labeling > completare Project name, Description

3. Selectare proiect
	* Selectare Projects (dreapta sus) > selectare proiect (ar trebui sa fie selectat deja by default)
	
4. Adaugare texte
	* Selectare Dataset > Actions > Import Dataset
	* File format:
		- TextFile - intreg fisierul este un document nou de adnotat
		- TextLine - fiecare linie este un document nou de adnotat	

5. Adnotare
	* Start Annotation (stanga sus)
	* Trecerea prin documente se face din dreapta sus sau folosind tastele stanga / dreapta
	* Pentru adaugarea unui label se selecteaza intai label-ul (din dreapta sau apasand cifra asociata) si apoi se selecteaza portiunea de text corespunzatoare
		! Nu sunt acceptate secvente pe mai multe randuri, asa ca adnotate separat, cu acelasi label
	* Documentul poate fi bifat ca adnotat din stanga sus sau folosind Enter
	* Labels:
        - 1 - action = secventa care denumeste o actiune ("se va citi", "se inlocuieste"); ar trebui sa contina toate cuvintele relevante (ex: "se prelungeste pana" / "se prelungeste cu")
		- 2 - old = secventa care denumeste partea relevanta din textul vechi
		- 3 - new = secventa care denumeste partea relevanta din textul nou