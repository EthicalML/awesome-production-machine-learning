[![Awesome](images/awesome.svg)](https://github.com/sindresorhus/awesome)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
![GitHub](https://img.shields.io/badge/Release-PROD-yellow.svg)
![GitHub](https://img.shields.io/badge/Languages-MULTI-blue.svg)
![GitHub](https://img.shields.io/badge/License-MIT-lightgrey.svg)
[![GitHub](https://img.shields.io/twitter/follow/axsaucedo.svg?label=Follow)](https://twitter.com/AxSaucedo/)
	

# Awesome machine learning operations

This repository contains a curated list of awesome open source libraries that will help you deploy, monitor, version and scale your machine learning.

## Quick links to sections in this page

| | | |
|-|-|-|
|[🔍 Explaining predictions & models](#1-explaining-black-box-models-and-datasets) |[🔏 Privacy preserving ML](#2-privacy-preserving-machine-learning) | [📜 Model & data versioning](#3-model-and-data-versioning)|
|[🏁 Model Orchestration](#4-model-deployment-and-orchestration-frameworks)|[🌀 Feature engineering](#5-feature-engineering-automation)|[🤖 Neural Architecture Search](#6-neural-architecture-search)|
| [📓 Reproducible Notebooks](#7-data-science-notebook-frameworks) | [📊 Visualisation frameworks](#8-industrial-strength-visualisation-libraries) | [🔠 Industry-strength NLP](#9-industrial-strength-nlp) |
| [🧵 Data pipelines & ETL](#10-data-pipeline-etl-frameworks) | [🗞️ Data storage](#11-data-storage-optimisation) | [📡 Functions as a service](#12-function-as-a-service-frameworks) |
| [🗺️ Computation distribution](#13-computation-load-distribution-frameworks) | [📥 Model serialisation](#14-model-serialisation-formats) | [🎁 Compiler optimisation](#15-compiler-optimisation-frameworks)  |
| [💸 Commercial ML](#16-commercial-data-science-platforms) | [💰 Commercial ETL](#17-commercial-etl-platforms)| |

## 10 Min Video Overview

<table>
  <tr>
    <td width="30%">
        This <a href="https://www.youtube.com/watch?v=e21fQtI5YlY">10 minute video</a> provides an overview of the motivations for machine learning operations as well as a high level overview on some of the tools in this repo.
    </td>
    <td width="70%">
        <a href="https://www.youtube.com/watch?v=e21fQtI5YlY"><img src="images/video.png"></a>
    </td>
  </tr>
</table>

## Want to receive recurrent updates on this repo and other advancements?

<table>
  <tr>
    <td width="30%">
         You can join the <a href="https://ethical.institute/mle.html">Machine Learning Engineer</a> newsletter. You will receive updates on open source frameworks, tutorials and articles curated by machine learning professionals.
    </td>
    <td width="70%">
        <a href="https://ethical.institute/mle.html"><img src="images/mleng.png"></a>
    </td>
  </tr>
</table>


# Main Content

## 1. Explaining Black Box Models and Datasets

* [XAI - eXplainableAI](https://github.com/EthicalML/xai) ![](https://img.shields.io/github/stars/EthicalML/XAI.svg?style=social) - An eXplainability toolbox for machine learning.
* [SHAP](https://github.com/slundberg/shap) ![](https://img.shields.io/github/stars/slundberg/shap.svg?style=social) - SHapley Additive exPlanations is a unified approach to explain the output of any machine learning model.
* [DeepLIFT](https://github.com/kundajelab/deeplift) ![](https://img.shields.io/github/stars/kundajelab/deeplift.svg?style=social) - Codebase that contains the methods in the paper ["Learning important features through propagating activation differences"](https://arxiv.org/abs/1704.02685). Here is the [slides](https://docs.google.com/file/d/0B15F_QN41VQXSXRFMzgtS01UOU0/edit?filetype=mspresentation) and the [video](https://vimeo.com/238275076) of the 15 minute talk given at ICML.
* [TreeInterpreter](https://github.com/andosa/treeinterpreter) ![](https://img.shields.io/github/stars/andosa/treeinterpreter.svg?style=social) - Package for interpreting scikit-learn's decision tree and random forest predictions. Allows decomposing each prediction into bias and feature contribution components as described in http://blog.datadive.net/interpreting-random-forests/. 
* [LIME](https://github.com/marcotcr/lime) ![](https://img.shields.io/github/stars/marcotcr/lime.svg?style=social) - Local Interpretable Model-agnostic Explanations for machine learning models.
* [ELI5](https://github.com/TeamHG-Memex/eli5) ![](https://img.shields.io/github/stars/TeamHG-Memex/eli5.svg?style=social) - "Explain Like I'm 5" is a Python package which helps to debug machine learning classifiers and explain their predictions.
* [Skater](https://github.com/datascienceinc/Skater) ![](https://img.shields.io/github/stars/datascienceinc/Skater.svg?style=social) - Skater is a unified framework to enable Model Interpretation for all forms of model to help one build an Interpretable machine learning system often needed for real world use-cases
* [themis-ml](https://github.com/cosmicBboy/themis-ml) ![](https://img.shields.io/github/stars/cosmicBboy/themis-ml.svg?style=social) - themis-ml is a Python library built on top of pandas and sklearnthat implements fairness-aware machine learning algorithms.
* [AI Fairness 360](https://github.com/IBM/AIF360) ![](https://img.shields.io/github/stars/IBM/AIF360.svg?style=social) - A comprehensive set of fairness metrics for datasets and machine learning models, explanations for these metrics, and algorithms to mitigate bias in datasets and models. 
* [casme](https://github.com/kondiz/casme) ![](https://img.shields.io/github/stars/kondiz/casme.svg?style=social) - Example of using classifier-agnostic saliency map extraction on ImageNet presented on the paper ["Classifier-agnostic saliency map extraction"](https://arxiv.org/abs/1805.08249).
* [ContrastiveExplanation (Foil Trees)](https://github.com/MarcelRobeer/ContrastiveExplanation) ![](https://img.shields.io/github/stars/MarcelRobeer/ContrastiveExplanation.svg?style=social) - Accompanying code for the paper "Contrastive Explanations with Local Foil Trees".
* [DeepVis Toolbox](https://github.com/yosinski/deep-visualization-toolbox) ![](https://img.shields.io/github/stars/yosinski/deep-visualization-toolbox.svg?style=social) - This is the code required to run the Deep Visualization Toolbox, as well as to generate the neuron-by-neuron visualizations using regularized optimization. The toolbox and methods are described casually [here](http://yosinski.com/deepvis) and more formally in this [paper](https://arxiv.org/abs/1506.06579).
* [FairML](https://github.com/adebayoj/fairml) ![](https://img.shields.io/github/stars/adebayoj/fairml.svg?style=social) - FairML is a python toolbox auditing the machine learning models for bias.
* [fairness](https://github.com/algofairness/fairness-comparison) ![](https://img.shields.io/github/stars/algofairness/fairness-comparison.svg?style=social) - This repository is meant to facilitate the benchmarking of fairness aware machine learning algorithms based on [this paper](https://arxiv.org/abs/1802.04422).
* [Integrated-Gradients](https://github.com/ankurtaly/Integrated-Gradients) ![](https://img.shields.io/github/stars/ankurtaly/Integrated-Gradients.svg?style=social) - This repository provideds code for implementing integrated gradients for networks with image inputs. 
* [LOFO Importance](https://github.com/aerdem4/lofo-importance) ![](https://img.shields.io/github/stars/aerdem4/lofo-importance.svg?style=social) - LOFO (Leave One Feature Out) Importance calculates the importances of a set of features based on a metric of choice, for a model of choice, by iteratively removing each feature from the set, and evaluating the performance of the model, with a validation scheme of choice, based on the chosen metric.
* [L2X](https://github.com/Jianbo-Lab/L2X) ![](https://img.shields.io/github/stars/Jianbo-Lab/L2X.svg?style=social) - Code for replicating the experiments in the paper ["Learning to Explain: An Information-Theoretic Perspective on Model Interpretation"](https://arxiv.org/pdf/1802.07814.pdf) at ICML 2018
* [Aequitas](https://github.com/dssg/aequitas) ![](https://img.shields.io/github/stars/dssg/aequitas.svg?style=social) - An open-source bias audit toolkit for data scientists, machine learning researchers, and policymakers to audit machine learning models for discrimination and bias, and to make informed and equitable decisions around developing and deploying predictive risk-assessment tools.
* [pyBreakDown](https://github.com/MI2DataLab/pyBreakDown) ![](https://img.shields.io/github/stars/MI2DataLab/pyBreakDown.svg?style=social) - A model agnostic tool for decomposition of predictions from black boxes. Break Down Table shows contributions of every variable to a final prediction. 
* [rationale](https://github.com/taolei87/rcnn/tree/master/code/rationale) ![](https://img.shields.io/github/stars/taolei87/rcnn.svg?style=social) - Code to implement learning rationales behind predictions with code for paper ["Rationalizing Neural Predictions"](https://github.com/taolei87/rcnn/tree/master/code/rationale)
* [Tensorflow's cleverhans](https://github.com/tensorflow/cleverhans) ![](https://img.shields.io/github/stars/tensorflow/cleverhans.svg?style=social) - An adversarial example library for constructing attacks, building defenses, and benchmarking both. A python library to benchmark system's vulnerability to [adversarial examples](http://karpathy.github.io/2015/03/30/breaking-convnets/)
* [tensorflow's lucid](https://github.com/tensorflow/lucid) ![](https://img.shields.io/github/stars/tensorflow/lucid.svg?style=social) - Lucid is a collection of infrastructure and tools for research in neural network interpretability.
* [tensorflow's Model Analysis](https://github.com/tensorflow/model-analysis) ![](https://img.shields.io/github/stars/tensorflow/model-analysis.svg?style=social) - TensorFlow Model Analysis (TFMA) is a library for evaluating TensorFlow models. It allows users to evaluate their models on large amounts of data in a distributed manner, using the same metrics defined in their trainer. 
* [Tensorboard's Tensorboard WhatIf](https://pair-code.github.io/what-if-tool/) ![](https://img.shields.io/github/stars/tensorflow/tensorboard.svg?style=social) - Tensorboard screen to analyse the interactions between inference results and data inputs.
* [Themis](https://github.com/LASER-UMASS/Themis) ![](https://img.shields.io/github/stars/LASER-UMASS/Themis.svg?style=social) - Themis is a testing-based approach for measuring discrimination in a software system.
* [anchor](https://github.com/marcotcr/anchor) ![](https://img.shields.io/github/stars/marcotcr/anchor.svg?style=social) - Code for the paper ["High precision model agnostic explanations"](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf), a model-agnostic system that explains the behaviour of complex models with high-precision rules called anchors.
* [woe](https://github.com/boredbird/woe) ![](https://img.shields.io/github/stars/boredbird/woe.svg?style=social) - Tools for WoE Transformation mostly used in ScoreCard Model for credit rating



## 2. Privacy Preserving Machine Learning
* [Tensorflow Privacy](https://github.com/tensorflow/privacy) ![](https://img.shields.io/github/stars/tensorflow/privacy.svg?style=social) - A Python library that includes implementations of TensorFlow optimizers for training machine learning models with differential privacy.
* [TF-Encrypted](https://github.com/mortendahl/tf-encrypted) ![](https://img.shields.io/github/stars/mortendahl/tf-encrypted.svg?style=social) - A Python library built on top of TensorFlow for researchers and practitioners to experiment with privacy-preserving machine learning.
* [PySyft](https://github.com/OpenMined/PySyft) ![](https://img.shields.io/github/stars/OpenMined/PySyft.svg?style=social) - A Python library for secure, private Deep Learning. PySyft decouples private data from model training, using Multi-Party Computation (MPC) within PyTorch.
* [Uber SQL Differencial Privacy](https://github.com/uber/sql-differential-privacy) ![](https://img.shields.io/github/stars/uber/sql-differential-privacy.svg?style=social) - Uber's open source framework that enforces differential privacy for general-purpose SQL queries.
* [Intel Homomorphic Encryption Backend](https://github.com/NervanaSystems/he-transformer) ![](https://img.shields.io/github/stars/NervanaSystems/he-transformer.svg?style=social) - The Intel HE transformer for nGraph is a Homomorphic Encryption (HE) backend to the Intel nGraph Compiler, Intel's graph compiler for Artificial Neural Networks.



## 3. Model and Data Versioning
* [Data Version Control (DVC)](https://github.com/iterative/dvc) ![](https://img.shields.io/github/stars/iterative/dvc.svg?style=social) - A git fork that allows for version management of models
* [ModelDB](https://mitdbg.github.io/modeldb/) ![](https://img.shields.io/github/stars/mitdbg/modeldb.svg?style=social) - Framework to track all the steps in your ML code to keep track of what version of your model obtained which accuracy, and then visualise it and query it via the UI
* [Pachyderm](https://github.com/pachyderm/pachyderm) ![](https://img.shields.io/github/stars/pachyderm/pachyderm.svg?style=social) - Open source distributed processing framework build on Kubernetes focused mainly on dynamic building of production machine learning pipelines - [(Video)](https://www.youtube.com/watch?v=LamKVhe2RSM)
* [steppy](https://github.com/neptune-ml/steppy) ![](https://img.shields.io/github/stars/neptune-ml/steppy.svg?style=social) - Lightweight, Python3 library for fast and reproducible machine learning experimentation. Introduces simple interface that enables clean machine learning pipeline design.
* [Quilt Data](https://github.com/quiltdata/quilt) ![](https://img.shields.io/github/stars/quiltdata/quilt.svg?style=social) - Versioning, reproducibility and deployment of data and models.
* [ModelChimp](https://github.com/ModelChimp/modelchimp/) ![](https://img.shields.io/github/stars/ModelChimp/modelchimp.svg?style=social) - Framework to track and compare all the results and parameters from machine learning models [(Video)](https://vimeo.com/271246650)
* [PredictionIO](https://github.com/apache/predictionio) ![](https://img.shields.io/github/stars/apache/predictionio.svg?style=social) - An open source Machine Learning Server built on top of a state-of-the-art open source stack for developers and data scientists to create predictive engines for any machine learning task
* [MLflow](https://github.com/mlflow/mlflow) ![](https://img.shields.io/github/stars/mlflow/mlflow.svg?style=social) - Open source platform to manage the ML lifecycle, including experimentation, reproducibility and deployment.
* [Sacred](https://github.com/IDSIA/sacred) ![](https://img.shields.io/github/stars/IDSIA/sacred.svg?style=social) - Tool to help you configure, organize, log and reproduce machine learning experiments.
* [Catalyst](https://github.com/catalyst-team/catalyst) ![](https://img.shields.io/github/stars/catalyst-team/catalyst.svg?style=social) - High-level utils for PyTorch DL & RL research. It was developed with a focus on reproducibility, fast experimentation and code/ideas reusing. 
* [FGLab](https://github.com/Kaixhin/FGLab) ![](https://img.shields.io/github/stars/Kaixhin/FGLab.svg?style=social) - Machine learning dashboard, designed to make prototyping experiments easier.
* [Studio.ML](https://github.com/studioml/studio) ![](https://img.shields.io/github/stars/studioml/studio.svg?style=social) - Model management framework which minimizes the overhead involved with scheduling, running, monitoring and managing artifacts of your machine learning experiments.
* [Flor](https://github.com/ucbrise/flor/blob/master/rtd/index.rst) ![](https://img.shields.io/github/stars/ucbrise/flor.svg?style=social) - Easy to use logger and automatic version controller made for data scientists who write ML code
* [D6tflow](https://github.com/d6t/d6tflow) ![](https://img.shields.io/github/stars/d6t/d6tflow.svg?style=social) - A python library that allows for building complex data science workflows on Python.

## 4. Model Deployment and Orchestration Frameworks
* [Seldon](https://github.com/SeldonIO/seldon-core) ![](https://img.shields.io/github/stars/SeldonIO/seldon-core.svg?style=social) - Open source platform for deploying and monitoring machine learning models in kubernetes - [(Video)](https://www.youtube.com/watch?v=pDlapGtecbY)
* [Redis-ML](https://github.com/RedisLabsModules/redis-ml) ![](https://img.shields.io/github/stars/RedisLabsModules/redis-ml.svg?style=social) - Module available from unstable branch that supports a subset of ML models as Redis data types
* [Model Server for Apache MXNet (MMS)](https://github.com/awslabs/mxnet-model-server) ![](https://img.shields.io/github/stars/awslabs/mxnet-model-server.svg?style=social) - A model server for Apache MXNet from Amazon Web Services that is able to run MXNet models as well as Gluon models (Amazon's SageMaker runs a custom version of MMS under the hood)
* [Tensorflow Serving](https://www.tensorflow.org/serving/) ![](https://img.shields.io/github/stars/tensorflow/serving.svg?style=social) - High-performant framework to serve Tensofrlow models via grpc protocol able to handle 100k requests per second per core
* [Clipper](https://github.com/ucbrise/clipper) ![](https://img.shields.io/github/stars/ucbrise/clipper.svg?style=social) - Model server project from Berkeley's Rise Rise Lab which includes a standard RESTful API and supports TensorFlow, Scikit-learn and Caffe models
* [DeepDetect](https://github.com/beniz/deepdetect) ![](https://img.shields.io/github/stars/beniz/deepdetect.svg?style=social) - Machine Learning production server for TensorFlow, XGBoost and Cafe models written in C++ and maintained by Jolibrain
* [MLeap](https://github.com/combust/mleap) ![](https://img.shields.io/github/stars/combust/mleap.svg?style=social) - Standardisation of pipeline and model serialization for Spark, Tensorflow and sklearn
* [OpenScoring](https://github.com/openscoring/openscoring) ![](https://img.shields.io/github/stars/openscoring/openscoring.svg?style=social) - REST web service for scoring PMML models built and maintained by OpenScoring.io
* [Open Platform for AI](https://github.com/Microsoft/pai) ![](https://img.shields.io/github/stars/Microsoft/pai.svg?style=social) - Platform that provides complete AI model training and resource management capabilities.
* [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html) - Model server created by NVIDIA that runs models in ONNX format, including frameworks such as TensorFlow and MATLAB
* [Kubeflow](https://github.com/kubeflow/kubeflow) ![](https://img.shields.io/github/stars/kubeflow/kubeflow.svg?style=social) - A cloud native platform for machine learning based on Google’s internal machine learning pipelines.
* [Polyaxon](https://github.com/polyaxon/polyaxon) ![](https://img.shields.io/github/stars/polyaxon/polyaxon.svg?style=social) - A platform for reproducible and scalable machine learning and deep learning on kubernetes. - [(Video)](https://www.youtube.com/watch?v=Iexwrka_hys)
* [Ray](https://github.com/ray-project/ray) ![](https://img.shields.io/github/stars/ray-project/ray.svg?style=social) - Ray is a flexible, high-performance distributed execution framework for machine learning ([VIDEO](https://www.youtube.com/watch?v=D_oz7E4v-U0))

## 5. Feature Engineering Automation
* [auto-sklearn](https://automl.github.io/auto-sklearn/stable/) ![](https://img.shields.io/github/stars/automl/auto-sklearn.svg?style=social) - Framework to automate algorithm and hyperparameter tuning for sklearn
* [TPOT](https://epistasislab.github.io/tpot/) ![](https://img.shields.io/github/stars/epistasislab/tpot.svg?style=social) - Automation of sklearn pipeline creation (including feature selection, pre-processor, etc)
* [tsfresh](https://github.com/blue-yonder/tsfresh) ![](https://img.shields.io/github/stars/blue-yonder/tsfresh.svg?style=social) - Automatic extraction of relevant features from time series
* [Featuretools](https://www.featuretools.com/) - An open source framework for automated feature engineering
* [Colombus](http://i.stanford.edu/hazy/victor/columbus/) - A scalable framework to perform exploratory feature selection implemented in R
* [automl](https://github.com/ClimbsRocks/automl) ![](https://img.shields.io/github/stars/ClimbsRocks/automl.svg?style=social) - Automated feature engineering, feature/model selection, hyperparam. optimisation


## 6. Neural Architecture Search
* [Neural Network Intelligence](https://github.com/Microsoft/nni) ![](https://img.shields.io/github/stars/Microsoft/nni.svg?style=social) - NNI (Neural Network Intelligence) is a toolkit to help users run automated machine learning (AutoML) experiments.
* [Autokeras](https://github.com/jhfjhfj1/autokeras) ![](https://img.shields.io/github/stars/jhfjhfj1/autokeras.svg?style=social) - AutoML library for Keras based on ["Auto-Keras: Efficient Neural Architecture Search with Network Morphism"](https://arxiv.org/abs/1806.10282).
* [ENAS-PyTorch](https://github.com/carpedm20/ENAS-pytorch) ![](https://img.shields.io/github/stars/carpedm20/ENAS-pytorch.svg?style=social) - Efficient Neural Architecture Search (ENAS) in PyTorch based [on this paper](https://arxiv.org/abs/1802.03268).
* [Neural Architecture Search with Controller RNN](https://github.com/titu1994/neural-architecture-search) ![](https://img.shields.io/github/stars/titu1994/neural-architecture-search.svg?style=social) - Basic implementation of Controller RNN from [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) and [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012).
* [ENAS via Parameter Sharing]() - Efficient Neural Architecture Search via Parameter Sharing by [authors of paper](https://arxiv.org/abs/1802.03268).
* [ENAS-Tensorflow](https://github.com/MINGUKKANG/ENAS-Tensorflow) ![](https://img.shields.io/github/stars/MINGUKKANG/ENAS-Tensorflow.svg?style=social) - Efficient Neural Architecture search via parameter sharing(ENAS) micro search Tensorflow code for windows user.

## 7. Data Science Notebook Frameworks
* [Jupyter Notebooks](http://jupyter.org/) - Web interface python sandbox environments for reproducible development
* [Stencila](https://github.com/stencila/stencila) ![](https://img.shields.io/github/stars/stencila/stencila.svg?style=social) - Stencila is a platform for creating, collaborating on, and sharing data driven content. Content that is transparent and reproducible.
* [RMarkdown](https://github.com/rstudio/rmarkdown) ![](https://img.shields.io/github/stars/rstudio/rmarkdown.svg?style=social) - The rmarkdown package is a next generation implementation of R Markdown based on Pandoc.
* [Hydrogen](https://atom.io/packages/hydrogen) - A plugin for ATOM that enables it to become a jupyter-notebook-like interface that prints the outputs directly in the editor.
* [H2O Flow](https://www.h2o.ai/download/) - Jupyter notebook-like inteface for H2O to create, save and re-use "flows"

## 8. Industrial Strength Visualization libraries
* [Plotly Dash](https://github.com/plotly/dash) ![](https://img.shields.io/github/stars/plotly/dash.svg?style=social) - Dash is a Python framework for building analytical web applications without the need to write javascript.
* [PDPBox](https://github.com/SauceCat/PDPbox) ![](https://img.shields.io/github/stars/SauceCat/PDPbox.svg?style=social) - This repository is inspired by ICEbox. The goal is to visualize the impact of certain features towards model prediction for any supervised learning algorithm. (now support all scikit-learn algorithms)
* [PyCEbox](https://github.com/AustinRochford/PyCEbox) ![](https://img.shields.io/stars/AustinRochford/PyCEbox.svg?style=social) - Python Individual Conditional Expectation Plot Toolbox
* [Plotly.py](https://github.com/plotly/plotly.py) ![](https://img.shields.io/github/stars/plotly/plotly.svg?style=social) - An interactive, open source, and browser-based graphing library for Python.
* [Pixiedust](https://github.com/pixiedust/pixiedust) ![](https://img.shields.io/github/stars/pixiedust/pixiedust.svg?style=social) - PixieDust is a productivity tool for Python or Scala notebooks, which lets a developer encapsulate business logic into something easy for your customers to consume.
* [ggplot2](https://github.com/tidyverse/ggplot2) ![](https://img.shields.io/github/stars/tidyverse/ggplot2.svg?style=social) - An implementation of the grammar of graphics for python. 
* [seaborn](https://github.com/mwaskom/seaborn) ![](https://img.shields.io/github/stars/mwaskom/seaborn.svg?style=social) - Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics.
* [Bokeh](https://github.com/bokeh/bokeh) ![](https://img.shields.io/github/stars/bokeh/bokeh.svg?style=social) - Bokeh is an interactive visualization library for Python that enables beautiful and meaningful visual presentation of data in modern web browsers.
* [matplotlib](https://github.com/matplotlib/matplotlib) ![](https://img.shields.io/github/stars/matplotlib/matplotlib.svg?style=social) - A Python 2D plotting library which produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. 
* [pygal](https://github.com/Kozea/pygal) ![](https://img.shields.io/github/stars/Kozea/pygal.svg?style=social) - pygal is a dynamic SVG charting library written in python
* [Geoplotlib](https://github.com/andrea-cuttone/geoplotlib) ![](https://img.shields.io/github/stars/andrea-cuttone/geoplotlib.svg?style=social) - geoplotlib is a python toolbox for visualizing geographical data and making maps
* [Missigno](https://github.com/ResidentMario/missingno) ![](https://img.shields.io/github/stars/ResidentMario/missingno.svg?style=social) - missingno provides a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset.
* [XKCD-style plots](http://jakevdp.github.io/blog/2013/07/10/XKCD-plots-in-matplotlib/) - An XKCD theme for matblotlib visualisations


## 9. Industrial strenght NLP
* [SpaCy](https://github.com/explosion/spaCy) ![](https://img.shields.io/github/stars/explosion/spaCy.svg?style=social) - Industrial-strength natural language processing library built with python and cython by the explosion.ai team.
* [Flair](https://github.com/zalandoresearch/flair) ![](https://img.shields.io/github/stars/zalandoresearch/flair.svg?style=social) - Simple framework for state-of-the-art NLP developed by Zalando which builds directly on PyTorch.
* [Wav2Letter++](https://code.fb.com/ai-research/wav2letter/) - A speech to text system developed by Facebook's FAIR teams.

## 10. Data Pipeline ETL Frameworks
* [Apache Airflow](https://airflow.apache.org/) - Data Pipeline framework built in Python, including scheduler, DAG definition and a UI for visualisation
* [Luigi](https://github.com/spotify/luigi) ![](https://img.shields.io/github/stars/spotify/luigi.svg?style=social) - Luigi is a Python module that helps you build complex pipelines of batch jobs, handling dependency resolution, workflow management, visualisation, etc
* [Genie](https://github.com/Netflix/genie) ![](https://img.shields.io/github/stars/Netflix/genie.svg?style=social) - Job orchestration engine to interface and trigger the execution of jobs from Hadoop-based systems
* [Oozie](http://oozie.apache.org/) - Workflow scheduler for Hadoop jobs
* [Apache Nifi](https://github.com/apache/nifi) ![](https://img.shields.io/github/stars/apache/nifi.svg?style=social) - Apache NiFi was made for dataflow. It supports highly configurable directed graphs of data routing, transformation, and system mediation logic.


## 11. Data Storage Optimisation
* [EdgeDB](https://edgedb.com/) - NoSQL interface for Postgres that allows for object interaction to data stored
* [BayesDB](http://probcomp.csail.mit.edu/software/bayesdb/) - Database that allows for built-in non-parametric Bayesian model discovery and queryingi for data on a database-like interface - [(Video)](https://www.youtube.com/watch?v=2ws84s6iD1o)
* [Apache Arrow](https://arrow.apache.org/) - In-memory columnar representation of data compatible with Pandas, Hadoop-based systems, etc
* [Apache Parquet](https://parquet.apache.org/) - On-disk columnar representation of data compatible with Pandas, Hadoop-based systems, etc
* [Apache Kafka](https://kafka.apache.org/) - Distributed streaming platform framework
* [ClickHouse](https://clickhouse.yandex/) - ClickHouse is an open source column oriented database management system supported by Yandex - [(Video)](https://www.youtube.com/watch?v=zbjub8BQPyE)
* [Alluxio](https://www.alluxio.org/docs/1.8/en/Overview.html) - A virtual distributed storage system that bridges the gab between computation frameworks and storage systems.


## 12. Function as a Service Frameworks
* [OpenFaaS](https://github.com/openfaas/faas) ![](https://img.shields.io/github/stars/openfaas/faas.svg?style=social) - Serverless functions framework with RESTful API on Kubernetes
* [Fission](https://github.com/fission/fission) ![](https://img.shields.io/github/stars/fission/fission.svg?style=social) - (Early Alpha) Serverless functions as a service framework on Kubernetes
* [Hydrosphere ML Lambda](https://github.com/Hydrospheredata/hydro-serving) ![](https://img.shields.io/github/stars/Hydrospheredata/hydro-serving.svg?style=social) - Open source model management cluster for deploying, serving and monitoring machine learning models and ad-hoc algorithms with a FaaS architecture
* [Hydrosphere Mist](https://github.com/Hydrospheredata/mist) ![](https://img.shields.io/github/stars/Hydrospheredata/mist.svg?style=social) - Serverless proxy for Apache Spark clusters
* [Apache OpenWhisk](https://github.com/apache/incubator-openwhisk) ![](https://img.shields.io/github/stars/apache/incubator-openwhisk.svg?style=social) - Open source, distributed serverless platform that executes functions in response to events at any scale. 

## 13. Computation load distribution frameworks
* [Hadoop Open Platform-as-a-service (HOPS)](https://www.hops.io/) - A multi-tenency open source framework with RESTful API for data science on Hadoop which enables for Spark, Tensorflow/Keras, it is Python-first, and provides a lot of features
* [PyWren](http://pywren.io) - Answer the question of the "cloud button" for python function execution. It's a framework that abstracts AWS Lambda to enable data scientists to execute any Pyhton function - [(Video)](https://www.youtube.com/watch?v=OskQytBBdJU)
* [NumPyWren](https://github.com/Vaishaal/numpywren) ![](https://img.shields.io/github/stars/Vaishaal/numpywren.svg?style=social) - Scientific computing framework build on top of pywren to enable numpy-like distributed computations
* [BigDL](https://bigdl-project.github.io/) - Deep learning framework on top of Spark/Hadoop to distribute data and computations across a HDFS system
* [Horovod](https://github.com/uber/horovod) ![](https://img.shields.io/github/stars/uber/horovod.svg?style=social) - Uber's distributed training framework for TensorFlow, Keras, and PyTorch
* [Apache Spark MLib](https://spark.apache.org/mllib/) - Apache Spark's scalable machine learning library in Java, Scala, Python and R
* [Dask](http://dask.pydata.org/en/latest/) - Distributed parallel processing framework for Pandas and NumPy computations - [(Video)](https://www.youtube.com/watch?v=RA_2qdipVng)


## 14. Model serialisation formats
* [ONNX](https://github.com/onnx/onnx) ![](https://img.shields.io/github/stars/onnx/onnx.svg?style=social) - Open Neural Network Exchange Format
* [Neural Network Exchange Format (NNEF)](https://www.khronos.org/nnef) - A standard format to store models across Torch, Caffe, TensorFlow, Theano, Chainer, Caffe2, PyTorch, and MXNet
* [PFA](http://dmg.org/pfa/index.html) - Created by the same organisation as PMML, the Predicted Format for Analytics is an emerging standard for statistical models and data transformation engines.
* [PMML](http://dmg.org/pmml/v4-3/GeneralStructure.html) - The Predictive Model Markup Language standard in XML - ([Video](https://www.youtube.com/watch?v=_5pZm2PZ8Q8))_
* [MMdnn](https://github.com/Microsoft/MMdnn) ![](https://img.shields.io/github/stars/Microsoft/MMdnn.svg?style=social) - Cross-framework solution to convert, visualize and diagnose deep neural network models. 
* [Java PMML API](https://github.com/jpmml) - Java libraries for consuming and producing PMML files containing models from different frameworks, including:
    * [sklearn2pmml](https://github.com/jpmml/jpmml-sklearn)
    * [pyspark2pmml](https://github.com/jpmml/pyspark2pmml)
    * [r2pmml](https://github.com/jpmml/r2pmml)
    * [sparklyr2pmml](https://github.com/jpmml/sparklyr2pmml)


## 15. Compiler optimisation frameworks
* [Numba](https://github.com/numba/numba) - A compiler for Python array and numerical functions

## 16. Commercial Data-science Platforms
* [cnvrg.io](https://cnvrg.io) - An end-to-end platform to manage, build and automate machine learning
* [Comet.ml](http://comet.ml) - Machine learning experiment management. Free for open source and students [(Video)](https://www.youtube.com/watch?v=xaybRkapeNE)
* [Skytree 16.0](http://skytree.net) - End to end machine learning platform [(Video)](https://www.youtube.com/watch?v=XuCwpnU-F1k)
* [Algorithmia](https://algorithmia.com/) - Cloud platform to build, deploy and serve machine learning models [(Video)](https://www.youtube.com/watch?v=qcsrPY0koyY)
* [y-hat](https://www.yhat.com/) - Deployment, updating and monitoring of predictive models in multiple languages [(Video)](https://www.youtube.com/watch?v=YiEjaWwzS_w)
* [Amazon SageMaker](https://aws.amazon.com/sagemaker/) - End-to-end machine learning development and deployment interface where you are able to build notebooks that use EC2 instances as backend, and then can host models exposed on an API
* [Google Cloud Machine Learning Engine](https://cloud.google.com/ml-engine/) - Managed service that enables developers and data scientists to build and bring machine learning models to production.
* [Microsoft Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) - Build, train, and deploy models from the cloud to the edge.
* [IBM Watson Machine Learning](https://www.ibm.com/cloud/machine-learning) - Create, train, and deploy self-learning models using an automated, collaborative workflow.
* [neptune.ml](https://neptune.ml) - community-friendly platform supporting data scientists in creating and sharing machine learning models. Neptune facilitates teamwork, infrastructure management, models comparison and reproducibility.
* [Datmo](https://datmo.com/) - Workflow tools for monitoring your deployed models to experiment and optimize models in production.
* [Valohai](https://valohai.com/) - Machine orchestration, version control and pipeline management for deep learning.
* [Dataiku](https://www.dataiku.com/) - Collaborative data science platform powering both self-service analytics and the operationalization of machine learning models in production.
* [MCenter](https://www.parallelm.com/product/) - MLOps platform automates the deployment, ongoing optimization, and governance of machine learning applications in production.
* [Skafos](https://metismachine.com/products/) - Skafos platform bridges the gap between data science, devops and engineering; continuous deployment, automation and monitoring.
* [SKIL](https://skymind.ai/platform) - Software distribution designed to help enterprise IT teams manage, deploy, and retrain machine learning models at scale.
* [MLJAR](https://mljar.com/) - Platform for rapid prototyping, developing and deploying machine learning models.
* [MissingLink](https://missinglink.ai/) - MissingLink helps data engineers streamline and automate the entire deep learning lifecycle.
* [DataRobot](https://www.datarobot.com/) - Automated machine learning platform which enables users to build and deploy machine learning models.
* [RiseML](https://riseml.com/) - Machine Learning Platform for Kubernetes: RiseML simplifies running machine learning experiments on bare metal and cloud GPU clusters of any size.
* [Datatron](https://datatron.com/) - Machine Learning Model Governance Platform for all your AI models in production for large Enterprises.

## 17. Commercial ETL Platforms
* [Talend Studio](https://www.talend.com/)
