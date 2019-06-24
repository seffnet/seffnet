## chem_sim_graph.pickle
a graph created from chemicals in SIDER database
Nodes: Chemicals
Edges: association (similarities)
## drugbank_pubchem_mapping.tsv
contains three columns (PubChemID, Smiles, DrugBankName)
## fullgraph.edgelist
a graph that contains chemicals from SIDER & DrugBank, side effects from SIDER and targets from DrugBank. The chemicals have similarities relationships.
## fullgraph_chem_sim.edgelist / fullgraph_chem_sim.pickle
a graph containing the chemicals from SIDER & DrugBank as nodes. the nodes are connected by their chemical similarities.
## fullgraph_nodes_mapping.tsv
maps the nodeID to its name (NodeID, NodeNamespace, NodeName)
## fullgraph_without_sim.edgelist/ fullgraph_without_sim.pickle
a graph that is similar to fullgraph.edgelist but without the chemical similarities relationships
## sider_graph.pickle
a graph created from SIDER database
