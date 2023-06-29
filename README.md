# ALLAN implementation for Indaco LegeAI

## Romanian disclaimer / Nota

> Acest produs a fost livrat si realizat in baza serviciilor de cercetare-inovare industrială conform contract de servicii nr. 9 din 01.11.2021 folosind modulele AI "ALLAN" aferente "TempRent" -  Proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și serviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, Contractul de finanțare nr. 2/221_ap3/24.06.2021.


## English disclaimer
> Copyright 2019-2021 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.
> NOTICE:  All information contained herein is, and remains the property of Knowledge Investment Group SRL.   The intellectual and technical concepts contained herein are proprietary to Knowledge Investment Group SRL and may be covered by Romanian and Foreign Patents, patents in process, and are protected by trade secret or copyright law. Dissemination of this information or reproduction of this material is strictly forbidden unless prior written permission is obtained from Knowledge Investment Group SRL.

## API documentation
Each system functionality is hosted by a microservice. All microservices are orchestrated by a gateway.

The single communication point is the gateway which can be accessed via `HTTP REST` requests (`POST` and `GET`).

* ### 1\. Endpoints

    * #### 1.1\. Run AI
     
       `POST http://195.60.78.150:5002/run`
       
       **Request body:**
        ```python
        {
            "SIGNATURE" : <String: mandatory>, # the name of the microservice
            "<PARAM1>"  : ..., # specific parameter of microservice - described in section 2 of the API documentation
            "<PARAM2>"  : ..., # specific parameter of microservice - described in section 2 of the API documentation
            ...                # specific parameter of microservice - described in section 2 of the API documentation
        }
        ```
        
        **Response if the AI runs successfully:**
        ```python
        {
            "call_id"    : <Integer>, # counter - the number of requests processed so far
            "signature"  : <String>, # worker signature - which worker resolved the input
            "<output_1>" : ..., # specific output of microservice - described in section 2 of the API documentation
            "<output_2>" : ..., # specific output of microservice - described in section 2 of the API documentation
            ...                 # specific output of microservice - described in section 2 of the API documentation
        }
        ```
        
        **Response if the AI encounters an error:**
        ```python
        {
            "call_id"   : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "ERROR" : <String> # error message that helps debugging
        }
        ```
        
        **Response if the signature is not started:**
        ```python
        {
            "ERROR": "Bad signature $(SIGNATURE). Available signatures: $(list_of_started_microservices)"
        }
        ```
    
    * #### 1.2\. Display microservices logs
    
        `POST http://195.60.78.150:5002/notifications`
        
        **Request body:**
        ```python
        {
            "SIGNATURE" : <String: mandatory> # the name of the microservice
        }
        ```
        
        **Response on success:**
        ```python
        {
            "1" : <List[dict]>, # List of messages reported at call_id=1; each message is a dictionary
            "2" : <List[dict]>, # List of messages reported at call_id=2; each message is a dictionary
            ...,
            "INIT"    : <List[dict]>, # List of messages reported at workers initialization; each message is a dictionary
            "GENERAL" : <List[dict]>  # List of messages reported at microservice initialization; each message is a dictionary
        }
        ```
        
        **Response if the signature is not started:**
        ```python
        {
            "ERROR": "Bad signature $(SIGNATURE). Available signatures: $(list_of_started_microservices)"
        }
        ```
  
    * #### 1.3\. Start server
    
        `POST http://195.60.78.150:5002/start_server`
        
        **Request body:**
        ```python
        {
            "SIGNATURE" : <String: mandatory> # the name of the microservice
        }
        ```
        
        **Response on success:**
        ```python
        {
            "MESSAGE": "OK."
        }
        ```
        
        **Response if microservice already started:**
        ```python
        {
            "ERROR" : "Signature $(SIGNATURE) already started"
        }
        ```
    
    * #### 1.4\. Kill server
    
        `POST http://195.60.78.150:5002/kill_server`
        
        **Request body:**
        ```python
        {
            "SIGNATURE" : <String: mandatory> # the name of the microservice
        }
        ```
        
        **Response on success:**
        ```python
        {
            "MESSAGE" : "OK. Killed PID=$(process_pid) with return_code $(return_code)."
        }
        ```
        
        **Response if the signature is not started:**
        ```python
        {
            "ERROR": "Bad signature $(SIGNATURE). Available signatures: $(list_of_started_microservices)>"
        }
        ```

* ### 2\. Developed microservices

    * #### 2.1\. get_sim - utilitary
    
        Given a word returns top n most semantically similar words in the vocabulary.
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_sim",
            "QUERY" : <String: mandatory>, # the word for which is called the microservice 
            "TOP_N" : <Integer: optional>  # default value 5
        }
        ```
        
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "results" : <List[string]> # List of most similar words
        }
        ```
        
        **Example:**
        ```python
        POST http://195.60.78.150:5002/run
        ----------------------------------
        {
            "SIGNATURE": "get_sim",
            "QUERY": "inginer",
            "TOP_N" : 3
        }
        
        Response json
        -------------
        {
            "call_id": 9,
            "results": ["constructor", "subinginer", "proiectant"],
            "signature": "GetSimWorker:0"
        }
        ```
       
    * #### 2.2\. get_aprox - utilitary
    
        Given a mispelled word returns top n most similar words in the vocabulary.
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_aprox",
            "QUERY" : <String: mandatory>, # the word for which is called the microservice
            "TOP_N" : <Integer: optional> # default value 1
        }
        ```
        
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "results" : <String> or <List[string]> # most similar word (if TOP_N=1) or list containing most similar words
        }
        ```
        
        **Example:**
        ```python
        POST http://195.60.78.150:5002/run
        ----------------------------------
        {
            "SIGNATURE": "get_aprox",
            "QUERY": "stlicla"
        }
        
        Response json
        -------------
        {
            "call_id": 13,
            "results": "sticla",
            "signature": "GetAproxWorker:2"
        }
        ```

    * #### 2.3\. get_tags - system functionality
        
        Given a document, returns top n associated tags and their scores
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_tags",
            "DOCUMENT" : <String: mandatory>, # document to be tested
            "TOP_N" : <Integer: optional> # default value 10
        }
        ```
        
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "results" : <List[[String, Integer]]> # [[tag1, score1], [tag2, score2], ...]
        }
        ```
        
        **Example:**
        ```python
        POST http://195.60.78.150:5002/run
        ----------------------------------
        {
            "SIGNATURE": "get_tags",
            "DOCUMENT": " Art. 20. - Jurisprudenţă, Reviste (10), Doctrină (4) (1) Publicitatea asigură opozabilitatea dreptului, actului, faptului, precum şi a oricărui alt raport juridic supus publicităţii, stabileşte rangul acestora şi, dacă legea prevede în mod expres, condiţionează constituirea sau efectele lor juridice. Reviste (12), Doctrină (1) (2) Între părţi sau succesorii lor, universali ori cu titlu universal, după caz, drepturile, actele sau faptele juridice, precum şi orice alte raporturi juridice produc efecte depline, chiar dacă nu au fost îndeplinite formalităţile de publicitate, afară de cazul în care prin lege se dispune altfel. Reviste (6), Doctrină (1) (3) Publicitatea nu validează dreptul, actul sau faptul supus ori admis la publicitate. Cu toate acestea, în cazurile şi condiţiile expres prevăzute de lege, ea poate produce efecte achizitive în favoarea terţilor dobânditori de bună-credinţă. Reviste (4), Doctrină (1)(4) Publicitatea nu întrerupe cursul prescripţiei extinctive, afară de cazul în care prin lege se dispune altfel. Doctrină (1)",
            "TOP_N": 1
        }
        
        Response json
        -------------
        {
            "call_id": 5,
            "results": [
                 ["administratie_publica", 0.07]
            ],
            "signature": "GetTagsWorker:0"
        }
        ```

    * #### 2.4\. get_qa - system functionality
    
        Given a natural language written question/sentence, returns top n associated tags and their scores
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_qa",
            "QUERY" : <String: mandatory>, # question to be tested>
            "TOP_N" : <Integer: optional> # default value 10
        }
        ```
        
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "results" : <List[[String, Integer]]> # [[tag1, score1], [tag2, score2], ...]
        }
        ```
        
        **Example:**
        ```python
        POST http://195.60.78.150:5002/run
        ----------------------------------
        {
            "SIGNATURE": "get_qa",
            "QUERY": "Care este regimul de tva intracomunitar si dubla taxare daca vreau sa cumpar o masina second hand din germania?",
            "TOP_N": 3
        }
        
        Response json
        -------------
        {
            "call_id": 5,
            "results": [
                 ["impozite_si_taxe", 0.356],
                 ["taxa_pe_valoare_adaugata", 0.098],
                 ["import_export", 0.04]
            ],
            "signature": "GetQAWorker:0"
        }
        ```
        
     * #### 2.5\. get_conf - system functionality
        Given a natural language written document, performs confidentialization of sensible information
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_conf",
            "DOCUMENT" : <String: mandatory>, # document to be confidentialized>
            "COMMANDS" : <List<dict>: optional> # human configured commands for helping and improving manually the endpoint performance 
        }
        ```
        
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "output" : <String> # the input document having confidentialized the sensible information,
            "position" : <List[[Integer, Integer, String]]> # [[pos_start_1, pos_end_1, confidential_categ_1], [pos_start_2, pos_end_2, confidential_categ_2], ...]
            "name_codes" : <Dictionary[String:String]> # [name1 : code1, name2 : code2, ...], the code assigned to each name
        }
        ```
        
        **Example:**
        ```python
        POST http://195.60.78.150:5002/run
        ----------------------------------
        {
            "SIGNATURE": "get_conf",
            "DOCUMENT": "Subsemnatul Popescu Costel, nascut la data 26.01.1976, domiciliat in Cluj, str. Cernauti, nr. 17-21, bl. K, parter, ap. 15 , declar pe propria raspundere ca sotia mea Alina Popescu, avand domiciliul flotant in Voluntari, str. Drumul Potcoavei nr 122, bl. B, sc. B, et. 1, ap 7, avand CI cu CNP 18634243332 nu detine proprietati."
        }
        
        Response json
        -------------
        {
            "call_id": 3,
            "ouptut" : "Subsemnatul A, nascut la data X, domiciliat in X. 1 , declar pe propria raspundere ca sotia mea B, avand domiciliul flotant in Voluntari, X, avand CI cu CNP X nu detine averi ilicite."
            "position": [
               [300, 313, "CNP"],
               [12, 31, "NUME"],
               [171, 185, "NUME"],
               [74, 123, "ADRESA"],
               [226, 282, "ADRESA"],
               [48, 58, "NASTERE"]
            ],
            "name_codes" : {
               "Popescu Costel" : "A",
               "Alina Popescu" : "B"
            }
            "signature": "GetConfWorker:1"
        }
        ```
     * #### 2.6\. get_sum2 - system functionality
        Given a natural language written document, performs summarization.
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_sum2",
            "DOCUMENT" : <String: mandatory>, # document to be summarized
        }
        ```

        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "output" : <String> # the input document having confidentialized the sensible information,
            "v0" : <List[String]>, # summarization v0 (January version)
            "v1" : <String>, # summarization v1
            "v2" : <String>, # summarization v2
            "v3" : <String>  # summarization v3
        }
        ```
        
     * #### 2.7\. get_mark - system functionality
        Given a natural language written query and a list of documents, chooses top documents that match with the query
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_mark",
            "QUERY" : <String: mandatory>, # natural language query
            "DOCUMENTS" : <List[String] : mandatory>, # list of documents
            "TOP_N" : <Integer : mandatory> # controlls the number of top results (top documents). if "TOP_N"=0, then all documents in the input will be sorted
        }
        ```
       
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "results" : <List[Integer]> # list with input documents indexes that match with the query (the length of this list is controlled by the "TOP_N" parameter; if it is 0, then it sorts all documents in input)
        }
        ```

     * #### 2.8\. get_merge - system functionality
        Given 2 paragraphs (passive and active) the endpoint merges them, returning the transformed paragraph
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_merge",
            "PASIV" : <String: mandatory>, # passive paragraph
            "ACTIV" : <String: mandatory>  # active paragraph
        }
        ```
       
        **Output:**
        ```python
        {
            "call_id" : <Integer>, # counter - the number of requests processed so far
            "signature" : <String>, # worker signature - which worker resolved the input,
            "success" : <Boolean> # True if the transformation was successful
            "action" :  <List[String]> # the list of identified actions; can be empty,            
            "old" :  <List[String]> # the list of phrases identified as Old; can be empty,             
            "new" :  <List[String]> # the list of phrases identified as New; can be empty,   
            "result" : <String> # the transformed paragraph; used only if success is True,
            "error": <String> # the type of error which occured; used only if success is False
        }
        ```
        
        **Errors:**
        ```python
        {
            "No Action identified." : no action could be identified,
            "Too many Actions identified. Can only handle a single Action." : several different types of actions were identified, but the model currently only handles single actions,
            "Incorrect number of Old and New entities identified." : the number of entities identified as Old and New does not match what was expected for the action,
            "Unknown action." : did not identify a known action,
            "Conditions for Old or New entities to apply Action not satisfied." : the entities identified as Old and New do not match the conditions required for the specific action
        }
        ```
        
        
        
        
* ### 3\. Microservices configuration

Each microservice can be configured in `config_gateway.txt`. The number of workers per each microservice is controlled with `NR_WORKERS`.


---


## Conda environments

* ### 1\. Deployment
   ```
   conda create -n allan_prod python=3.8 pip
   pip install tensorflow==2.5 pandas matplotlib psutil tqdm shapely seaborn scikit-learn scikit-image dropbox gensim transformers flask unidecode phonenumbers pyodbc
   pip install -U pip setuptools wheel
   pip install -U spacy
   ```


* ### 2\. Windows development (GPU)
   ```
   conda create -n allan_dev python=3.8 pip
   conda install -c anaconda tensorflow-gpu=2.5
   pip install pandas matplotlib psutil tqdm shapely seaborn scikit-learn scikit-image dropbox gensim transformers flask pyodbc datasets
   ```
