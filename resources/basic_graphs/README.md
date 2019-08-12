# Basic graphs

The graphs in this folder were created using databases.

## drugbank_graph

This graph was created using DrugBank database (https://drugbank.ca).

It contains chemicals and proteins as nodes, and relations between chemicals and proteins as edges.</br>

- Chemicals (pubchem.compound): 6386
- Proteins (uniprot): 4049

## sider_graph

This graph was created using SIDER database (http://sideeffects.embl.de/).

It contains chemicals and side effects as nodes, and the relations between chemicals and side effects as edges.

- Chemicals (pubchem.compound): 1507
- Side Effects / Indiations (umls): 6990

## fullgraph_without_sim

This graph is a combination of both SIDER and DrugBank graphs.

The chemicals in both graphs were mapped using SMILES and the drugbank_pubchem_mapping file in the mapping folder

- Chemicals: 7274
- Proteins: 4049
- Side effects: 6990
