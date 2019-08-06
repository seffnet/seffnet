# Basic graphs
The graphs in this folder were created using databases. </br>
Related Notebook: https://github.com/AldisiRana/SE_KGE/blob/master/notebooks/creating_graphs.ipynb
## drugbank_graph
This graph was created using DrugBank database (https://drugbank.ca) </br>
It contains chemicals and proteins as nodes, and relations between chemicals and proteins as edges.</br>
Number of chemicals: 4958 </br>
Number of proteins: 2559 
## sider_graph
This graph was created using SIDER database (http://sideeffects.embl.de/) </br>
It contains chemicals and side effects as nodes, and the relations between chemicals and side effects as edges. </br>
Number of chemicals: 1507 </br>
Number of side effects: 6990 
## fullgraph_without_sim
This graph is a combination of both sider and drugbank graphs. The chemicals in both graphs were mapped using SMILES and the drugbank_pubchem_mapping file in the mapping folder</br>
Number of chemicals: 4743 </br>
Number of proteins: 1997 </br>
Number of side effects: 6990


