SEffNet |build|
===============
Structure
---------
- ``notebooks``: Notebooks that were used during the preprocessing and creation of graphs, and the training and evaluation of models
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

Web Application
~~~~~~~~~~~~~~~
The web application allows users to get results from the model programmatically.
Run with:

.. code-block:: bash

    $ seffnet web --host localhost --port 5000

- A user interface can be found at http://localhost:5000
- A swagger UI can be found at http://localhost:5000/apidocs

.. |build| image:: https://travis-ci.com/AldisiRana/SE_KGE.svg?branch=master
    :target: https://travis-ci.com/AldisiRana/SE_KGE
    :alt: Development Build Status
