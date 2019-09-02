SEffNet |build|
===============
Structure
---------
- ``notebooks``: Notebooks that were used for training and evaluation of models, and interpertation of prediction model
- ``resources``: The graphs and materials that are used for training and testing

Installation
------------
``se_kge`` can be installed on python37+ from the latest code on `GitHub <https://github.com/AldisiRana/SE_KGE>`_ with:

.. code-block:: sh

    $ pip install git+https://github.com/AldisiRana/SE_KGE.git

Usage
-----
Using the predictive model
~~~~~~~~~~~~~~~~~~~~~~~~~~
If you've installed ``seffnet`` locally, you can use the default model from the GitHub repository with:

.. code-block:: python

    from se_kge.default_predictor import predictor
    
    # Find new relations for a given entity based on its CURIE
    results = predictor.find_new_relations(curie='pubchem.compound:85')
    ...   

Optimizing hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~
Network representation learning models can be optimized with:

.. code-block:: bash

    $ seffnet optimize --input-path ./resources/chemsim_50_graphs/fullgraph_with_chemsim_50.edgelist --method node2vec
    
    
CLI Options:

- --input-path, input graph file. Only accepted edgelist format. If training-path and testing-path are not specified, the input graph will be split randomly.
- --method, the NRL method to train the model. Choices: node2vec, DeepWalk, HOPE, GraRep, LINE, SDNE.
- --training-path, training graph file. Only accepted edgelist format.
- --testing-path, testing graph file. Only accepted edgelist format.
- --trials, the number of trials done to optimize hyperparameters. Default=50
- --dimensions-range, the range of dimensions to be optimized. Default=100-300
- --storage, SQL connection string for study database. Example: sqlite:///optuna.db
- --name, name for the study
- -o, --output, Output study summary
- --seed, default is a random number between 1 and 10000000

Web Application
~~~~~~~~~~~~~~~
The web application allows users to get results from the model programmatically. Make 
sure the extra dependencies have been installed as well using the `[web]`

.. code-block:: sh

    $ pip install git+https://github.com/AldisiRana/SE_KGE.git[web]

Run with:

.. code-block:: bash

    $ seffnet web --host localhost --port 5000

- A user interface can be found at http://localhost:5000
- A swagger UI can be found at http://localhost:5000/apidocs

As an example, you can check the chemicals predicted to interact
with HDAC6 at http://localhost:5000/predict/uniprot:Q9UBN7?results_type=chemical.

.. |build| image:: https://travis-ci.com/AldisiRana/SE_KGE.svg?branch=master
    :target: https://travis-ci.com/AldisiRana/SE_KGE
    :alt: Development Build Status
