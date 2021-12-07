# ALLAN implementation for Indaco LegeAI

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

* ### 3\. Microservices configuration

Each microservice can be configured in `config_gateway.txt`. The number of workers per each microservice is controlled with `NR_WORKERS`.


---


## Conda environments

* ### 1\. Deployment
   ```
   conda create -n allan1 python=3.8 cudatoolkit=10.1 scikit-learn python-Levenshtein pandas pyodbc tqdm flask psutil pytorch -c pytorch
   pip install tensorflow-gpu==2.3.0
   pip install gensim
   ```


* ### 2\. Dev Windows including GUI
   ```
   conda create -n allan_dev tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0 tensorflow-hub spyder==4.2.5 pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox ipykernel=6.2.0 python-Levenshtein pyodbc flask pytorch=1.8 -c pytorch 
   conda install numpy=1.18.5
   pip install gensim=4.1.2
   ```

* ### 3\. Ubuntu - OPTIONAL
   ```
   conda create -n allan python=3.8 cudatoolkit=10.1  pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox python-Levenshtein pyodbc 
   conda activate allan
   pip install tensorflow-gpu==2.3
   pip install gensim
   pip install -U numpy==1.18.5
   ```
