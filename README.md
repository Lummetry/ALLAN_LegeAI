# ALLAN implementation for Indaco LegeAI

### Deployment
```
conda create -n allan1 tensorflow-gpu scikit-learn gensim python-Levenshtein pandas pyodbc flask pytorch -c pytorch
```


### Dev Windows including GUI
```
conda create -n allan_dev tensorflow-gpu=2.3 tensorflow=2.3=mkl_py38h1fcfbd6_0 tensorflow-hub gensim spyder==4.2.5 pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox ipykernel=6.2.0 python-Levenshtein pyodbc flask pytorch=1.8 -c pytorch 
```

##### Ubuntu - OPTIONAL
```
conda create -n allan python=3.8 cudatoolkit=10.1  pandas matplotlib psutil tqdm shapely seaborn scikit-learn dropbox python-Levenshtein pyodbc 
conda activate allan
pip install tensorflow-gpu==2.3
pip install gensim
pip install -U numpy==1.18.5
```