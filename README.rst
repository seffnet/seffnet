SEffNet |build|
===============
SEffNet (**S**\ide **Eff**\ect **Net**\work embeddings)  is a tool that optimizes, trains, and evaluates predictive models for biomedical networks that contain drug-, target- and side effect-information using different network representation learning methods in an attempt to understand the causes of side effects.

This package was developed during the `master's thesis <https://github.com/aldisirana/masters_thesis>`_
of `Rana Aldisi <https://github.com/aldisirana>`_.

Structure
---------
- ``notebooks``: Notebooks that were used for training and evaluation of models, and interpertation of prediction model
- ``resources``: The graphs and materials that are used for training and testing

Installation
------------
``seffnet`` can be installed on python37+ from the latest code on `GitHub <https://github.com/seffnet/seffnet>`_ with:

.. code-block:: sh

    $ pip install git+https://github.com/seffnet/seffnet.git

Usage
-----
Using the predictive model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you've installed ``seffnet`` locally, you can use the default model from the GitHub repository with:

.. code-block:: python

    from seffnet.default_predictor import predictor
    
    # Find new relations for a given entity based on its CURIE
    results = predictor.find_new_relations(curie='pubchem.compound:5095')
    ...   

You can use the default model in the CLI:

.. code-block:: bash

    $ seffnet predict pubchem.compound:5095

You can predict on new chemicals via their SMILES strings based on their similarity
to chemicals included in the network. Warning: we haven't benchmarked how well this
actually works yet.

.. code-block:: bash

    $ seffnet predictc "C1=CC=C(C=C1)C2=CC=C(C=C2)CCO"

Rebuilding the resources
~~~~~~~~~~~~~~~~~~~~~~~~~~
You can rebuild all the graphs and maps created for this project by running the following:

.. code-block:: bash

    $ seffnet rebuild
    
Note that you need to have RDKit package and environment to be able to run this command


Model training and evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can train an NRL model using the following:

.. code-block:: bash

    $ seffnet train --input-path ./resources/basic_graphs/fullgraph_with_chemsim.edgelist --evaluation --method node2vec
    
- For further CLI options and parameters use --help, -h

Optimizing hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~
Network representation learning models can be optimized with:

.. code-block:: bash

    $ seffnet optimize --input-path ./resources/basic_graphs/fullgraph_with_chemsim.edgelist --method node2vec
    
    
- For further CLI options and parameters use --help, -h

Web Application
~~~~~~~~~~~~~~~
The web application allows users to get results from the model programmatically. Make 
sure the extra dependencies have been installed as well using the `[web]` extra.
Unfortunately, this doesn't work when installing directly from GitHub, so see the
``setup.cfg`` for the Flask dependencies.

.. code-block:: sh

    $ pip install -e .[web]

Run development server with:

.. code-block:: bash

    $ seffnet web --host localhost --port 5000

Run through docker with:

.. code-block:: bash

    $ docker-compose up

- A user interface can be found at http://localhost:5000
- An auto-generated swagger UI can be found at http://localhost:5000/apidocs

As an example, you can check the chemicals predicted to interact
with HDAC6 at http://localhost:5000/predict/uniprot:Q9UBN7?results_type=chemical.

.. |build| image:: https://travis-ci.com/seffnet/seffnet.svg?branch=master
    :target: https://travis-ci.com/seffnet/seffnet
    :alt: Development Build Status
