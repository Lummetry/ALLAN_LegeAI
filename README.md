# ALLAN implementation for Indaco LegeAI

### API documentation
Each system functionality is hosted in a microservice. All microservices are orchestrated by a gateway.
The single communication point is the gateway which can be accessed via HTTP REST requests (POST and GET).

1. Endpoints

    1. Run AI
     
       `POST http://195.60.78.150:5002/run`
       
       **Request body:**
        ```
        {
            "SIGNATURE" : "<the name of the microservice: Mandatory>",
            "<PARAM1>" : ..., # specific parameter of microservice - described in section 2 of the API documentation
            "<PARAM2>" : ..., # specific parameter of microservice - described in section 2 of the API documentation
            ...               # specific parameter of microservice - described in section 2 of the API documentation
        }
        ```
        
        **Response if the AI runs successfully:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "<output_1>" : ..., # specific output of microservice - described in section 2 of the API documentation
            "<output_2>" : ..., # specific output of microservice - described in section 2 of the API documentation
            ...                 # specific output of microservice - described in section 2 of the API documentation
        }
        ```
        
        **Response if the AI encounters an error:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "ERROR" : "<error message that helps debugging>"
        }
        ```
        
        **Response if the signature is not correct:**
        ```
        {
            "ERROR": "Bad signature <SIGNATURE>. Available signatures: <List of started microservices>"
        }
        ```
    
    2. Display microservices logs
    
        `POST http://195.60.78.150:5002/notifications`
        
        **Request body:**
        ```
        {
            "SIGNATURE" : "<the name of the microservice: Mandatory>"
        }
        ```
        
        **Response on success:**
        ```
        {
            "1" : [List of messages reported at call_id=1],
            "2" : [List of messages reported at call_id=2],
            ...,
            "INIT"    : [List of messages reported at workers initialization],
            "GENERAL" : [List of messages reported at microservice initialization]
        }
        ```
        
        **Response if the signature is not started:**
        ```
        {
            "ERROR": "Bad signature <SIGNATURE>. Available signatures: <List of started microservices>"
        }
        ```
  
    3. Start server
    
        `POST http://195.60.78.150:5002/start_server`
        
        **Request body:**
        ```
        {
            "SIGNATURE" : "<the name of the microservice: Mandatory>"
        }
        ```
        
        **Response on success:**
        ```
        {
            "MESSAGE": "OK."
        }
        ```
        
        **Response if microservice already started:**
        ```
        {
            "ERROR" : "Signature <SIGNATURE> already started"
        }
        ```
    
    4. Kill server
    
        `POST http://195.60.78.150:5002/kill_server`
        
        **Request body:**
        ```
        {
            "SIGNATURE" : "<the name of the microservice: Mandatory>"
        }
        ```
        
        **Response on success:**
        ```
        {
            "MESSAGE" : 'OK. Killed PID=<process_pid> with return_code <return_code>.
        }
        ```
        
        **Response if the signature is not started:**
        ```
        {
            "ERROR": "Bad signature <SIGNATURE>. Available signatures: <List of started microservices>"
        }
        ```

2. Developed microservices

    1. get_sim - utilitary
    
        Given a word returns the most semantically similar words in the vocabulary.
        
        **Input parameters:**
        ```
        {
            "SIGNATURE" : "get_sim",
            "QUERY" : "<word>",
            "TOP_N" : <Integer : optional> # default value 5
        }
        ```
        
        **Output:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "results" : [List of most similar words]
        }
        ```
       
    2. get_aprox - utilitary
    
        Given a mispelled word returns the most similar words in the vocabulary.
        
        **Input parameters:**
        ```
        {
            "SIGNATURE" : "get_aprox",
            "QUERY" : "<word>",
            "TOP_N" : <Integer : optional> # default value 1
        }
        ```
        
        **Output:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "results" : [List of most similar words]
        }
        ```

    3. get_tags - system functionality
        
        Given a document, returns top n associated tags and their scores
        
        **Input parameters:**
        ```
        {
            "SIGNATURE" : "get_tags",
            "DOCUMENT" : "<string=document to be tested>",
            "TOP_N" : <Integer: optional> # default value 10
        }
        ```
        
        **Output:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "results" : [
                [tag1, score1],
                [tag2, score2],
                ...
            ]
        }
        ```

    4. get_qa - system functionality
    
        Given a natural language written question/sentence, returns top n associated tags and their scores
        
        **Input parameters:**
        ```
        {
            "SIGNATURE" : "get_tags",
            "QUERY" : "<string=question to be tested>",
            "TOP_N" : <Integer: optional> # default value 10
        }
        ```
        
        **Output:**
        ```
        {
            "call_id" : ..., # counter - the number of requests processed so far
            "signature" : ..., # worker signature - which worker resolved the input,
            "results" : [
                [tag1, score1],
                [tag2, score2],
                ...
            ]
        }
        ```

3. Microservices configuration

Each microservice can be configured in `config_gateway.txt`. The number of workers per each microservice is controlled with `NR_WORKERS`.





### Deployment
```
conda create -n allan1 python=3.8 cudatoolkit=10.1 scikit-learn python-Levenshtein pandas pyodbc tqdm flask psutil pytorch -c pytorch
pip install tensorflow-gpu==2.3.0
pip install gensim
```


### Dev Windows including GUI
```
conda create -n allan_dev tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0 tensorflow-hub spyder==4.2.5 pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox ipykernel=6.2.0 python-Levenshtein pyodbc flask pytorch=1.8 -c pytorch 
conda install numpy=1.18.5
pip install gensim=4.1.2
```

##### Ubuntu - OPTIONAL
```
conda create -n allan python=3.8 cudatoolkit=10.1  pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox python-Levenshtein pyodbc 
conda activate allan
pip install tensorflow-gpu==2.3
pip install gensim
pip install -U numpy==1.18.5
```