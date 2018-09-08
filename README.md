# Machine Learning Operations (ML + DevOps = MLOps)

## What is Machine Learning Operations (MLOps)?

Machine Learning Operations involves the infrastructure required to scale your ML capabilities.

I focus on two main principles when discussing machine learning operations:
* **Reproducibility**: Model & Data Versioning
* **Orchestration**: Model Deployment Orchestration

I cover the motivations & concepts in my talk at the 2018 EuroSciPy on [Scalable Data Science: The State of MLOps in 2018](https://axsauze.github.io/scalable-data-science/#/). Machine learning operations can be seen in a very high level form in the diagram below:

![](images/mlops1.png)


## Overview of this repo

* This repository contains a curated list of awesome resources that will help you kick-start or enhance your machine learning operations
* Machine Learning Operations involve everything that is required to serve your ML, including deploying, monitoring, scaling, versioning, etc

## Contents


### Model Standard Formats
* [ONNX - Open Neural Network Exchange Format](https://github.com/onnx/onnx)
* [PMLL - The Predictive Model Markup Language standard in XML](http://dmg.org/pmml/v4-3/GeneralStructure.html) - ([Video](https://www.youtube.com/watch?v=_5pZm2PZ8Q8))_
* [Java PMML API - Java libraries for consuming and producing PMML files containing models from different frameworks](https://github.com/jpmml), including:
    * [sklearn2pmml](https://github.com/jpmml/jpmml-sklearn)
    * [pyspark2pmml](https://github.com/jpmml/pyspark2pmml)
    * [r2pmml](https://github.com/jpmml/r2pmml)
    * [sparklyr2pmml](https://github.com/jpmml/sparklyr2pmml)

### Model Versioning
* [Data Version Control (DVC) - A git fork that allows for version management of models](https://dvc.org/)
* [ModelDB - Framework to track all the steps in your ML code to keep track of what version of your model obtained which accuracy, and then visualise it and query it via the UI](https://mitdbg.github.io/modeldb/)
* [Pachyderm - Open source distributed processing framework build on Kubernetes focused mainly on dynamic building of production machine learning pipelines](https://github.com/pachyderm/pachyderm) - [(Video)](https://www.youtube.com/watch?v=LamKVhe2RSM&t=1167s)
* [Jupyter Notebooks - Web interface python sandbox environments for reproducible development](http://jupyter.org/)
* [H2O Flow - Jupyter notebook-like inteface for H2O to create, save and re-use "flows"](https://www.h2o.ai/download/)


### Data Storage / Standardisation
* [EdgeDB - NoSQL interface for Postgres that allows for object interaction to data stored](https://edgedb.com/)
* [BayesDB - Database that allows for built-in non-parametric Bayesian model discovery and queryingi for data on a database-like interface](http://probcomp.csail.mit.edu/bayesdb/) - [(Video)](https://www.youtube.com/watch?v=2ws84s6iD1o)
* [Apache Arrow - In-memory columnar representation of data compatible with Pandas, Hadoop-based systems, etc](https://arrow.apache.org/)
* [Apache Parquet - On-disk columnar representation of data compatible with Pandas, Hadoop-based systems, etc](https://parquet.apache.org/)
* [Kafka]()

### Feature Engineering Automation
* [auto-sklearn - Framework to automate algorithm and hyperparameter tuning for sklearn](https://automl.github.io/auto-sklearn/stable/)
* [TPOT - Automation of sklearn pipeline creation (including feature selection, pre-processor, etc)](https://epistasislab.github.io/tpot/)
* [tsfresh - Automatic extraction of relevant features from time series](https://github.com/blue-yonder/tsfresh)
* [Featuretools - An open source framework for automated feature engineering](https://www.featuretools.com/)
* [Colombus - A scalable framework to perform exploratory feature selection implemented in R](http://i.stanford.edu/hazy/victor/columbus/)
* [automl - Automated feature engineering, feature/model selection, hyperparam. optimisation](https://github.com/ClimbsRocks/automl)


### Model Deployment Frameworks
* [Seldon - Open source platform for deploying and monitoring machine learning models in kubernetes](https://github.com/SeldonIO/seldon-core) - [(Video)](https://www.youtube.com/watch?v=pDlapGtecbY)
* [Redis-ML - Module available from unstable branch that supports a subset of ML models as Redis data types](https://github.com/RedisLabsModules/redis-ml)
* [Model Server for Apache MXNet (MMS) - A model server for Apache MXNet from Amazon Web Services that is able to run MXNet models as well as Gluon models (Amazon's SageMaker runs a custom version of MMS under the hood)](https://github.com/awslabs/mxnet-model-server)
* [Tensorflow Serving - High-performant framework to serve Tensofrlow models via grpc protocol able to handle 100k requests per second per core](https://www.tensorflow.org/serving/)
* [Clipper - Model server project from Berkeley's Rise Rise Lab which includes a standard RESTful API and supports TensorFlow, Scikit-learn and Caffe models](http://clipper.ai/)
* [DeepDetect - Machine Learning production server for TensorFlow, XGBoost and Cafe models written in C++ and maintained by Jolibrain](https://github.com/beniz/deepdetect)
* [MLeap - Standardisation of pipeline and model serialization for Spark, Tensorflow and sklearn](https://github.com/combust/mleap)
* [TensorRT - Model server created by NVIDIA that runs models in ONNX format, including frameworks such as TensorFlow and MATLAB](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html)
* [OpenScoring - REST web service for scoring PMML models built and maintained by OpenScoring.io](https://github.com/openscoring/openscoring)

### Distributing Computation Load
* [Dask - Distributed parallel processing framework for Pandas and NumPy computations](http://dask.pydata.org/en/latest/) - [(Video)](https://www.youtube.com/watch?v=RA_2qdipVng)
* [PyWren - Answer the question of the "cloud button" for python function execution. It']s a framework that abstracts AWS Lambda to enable data scientists to execute any Pyhton function](http://pywren.io) - [(Video)](https://www.youtube.com/watch?v=OskQytBBdJU)

### Data Pipeline Frameworks
* [Apache Airflow - Data Pipeline framework built in Python, including scheduler, DAG definition and a UI for visualisation](https://airflow.apache.org/)
* [Luigi - Luigi is a Python module that helps you build complex pipelines of batch jobs, handling dependency resolution, workflow management, visualisation, etc](https://github.com/spotify/luigi)
* [Genie - Job orchestration engine to interface and trigger the execution of jobs from Hadoop-based systems](https://github.com/Netflix/genie)
* [Oozie - Workflow scheduler for Hadoop jobs](http://oozie.apache.org/)

### Data Streaming Frameworks
* [Apache Kafka - Distributed streaming platform framework](https://kafka.apache.org/)

### Function as a Service Frameworks
* [OpenFaaS - Serverless functions framework with RESTful API on Kubernetes](https://github.com/openfaas/faas)
* [Fission - (Early Alpha) Serverless functions as a service framework on Kubernetes](https://github.com/fission/fission)
* [Hydrosphere ML Lambda - Open source model management cluster for deploying, serving and monitoring machine learning models and ad-hoc algorithms with a FaaS architecture](https://github.com/Hydrospheredata/hydro-serving)
* [Hydrosphere Mist - Serverless proxy for Apache Spark clusters](https://github.com/Hydrospheredata/mist)

### Optimization of Computaiton
* [Numba - A compiler for Python array and numerical functions](https://github.com/numba/numba)

### Machine learning resource management
### Monitoring
### Configuration

### Commercial Data-science Platforms
* [Skytree 16.0 - End to end machine learning platform](http://skytree.net) [(Video)](https://www.youtube.com/watch?v=XuCwpnU-F1k)
* [Algorithmia - Cloud platform to build, deploy and serve machine learning models](https://algorithmia.com/) [(Video)](https://www.youtube.com/watch?v=qcsrPY0koyY)
* [y-hat - Deployment, updating and monitoring of predictive models in multiple languages](https://www.yhat.com/)[(Video)](https://www.youtube.com/watch?v=YiEjaWwzS_w)
* [Amazon SageMaker - End-to-end machine learning development and deployment interface where you are able to build notebooks that use EC2 instances as backend, and then can host models exposed on an API]

### Commercial ETL Platforms
* Talend Studio


