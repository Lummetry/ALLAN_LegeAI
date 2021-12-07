# ALLAN implementation for Indaco LegeAI

## API documentation
Each system functionality is hosted in a microservice. All microservices are orchestrated by a gateway.
The single communication point is the gateway which can be accessed via HTTP REST requests (POST and GET).

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
        
        **Response if the signature is not correct:**
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
            "results" : <List[string]> # List of most similar words
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

    * #### 2.4\. get_qa - system functionality
    
        Given a natural language written question/sentence, returns top n associated tags and their scores
        
        **Input parameters:**
        ```python
        {
            "SIGNATURE" : "get_tags",
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

* ### 3\. Microservices configuration

Each microservice can be configured in `config_gateway.txt`. The number of workers per each microservice is controlled with `NR_WORKERS`.


---


## Deployment
```
conda create -n allan1 python=3.8 cudatoolkit=10.1 scikit-learn python-Levenshtein pandas pyodbc tqdm flask psutil pytorch -c pytorch
pip install tensorflow-gpu==2.3.0
pip install gensim
```


## Dev Windows including GUI
```
conda create -n allan_dev tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0 tensorflow-hub spyder==4.2.5 pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox ipykernel=6.2.0 python-Levenshtein pyodbc flask pytorch=1.8 -c pytorch 
conda install numpy=1.18.5
pip install gensim=4.1.2
```

## Ubuntu - OPTIONAL
```
conda create -n allan python=3.8 cudatoolkit=10.1  pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox python-Levenshtein pyodbc 
conda activate allan
pip install tensorflow-gpu==2.3
pip install gensim
pip install -U numpy==1.18.5
```
