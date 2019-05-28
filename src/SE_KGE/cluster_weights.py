import pandas as pd
import pybel
import networkx as nx
import tqdm

full_graph = pybel.from_pickle('/home/bio/groupshare/rana/fullgraph_without_sim.pickle')
cluster_df = pd.read_csv('/home/bio/groupshare/rana/src/SE_KGE/resources/Clustered_chemicals.csv')

clusters_dict = {i : cluster_df['PubchemID'].loc[cluster_df['Cluster'] == i].tolist()
                for i in range(1,cluster_df.Cluster.nunique()+1)}
  
subgraphs_dict = {}
for cluster, chemicals in clusters_dict.items():
    chemicals_subgraph = []
    for chemical in chemicals:
        chemical = pybel.dsl.Abundance(namespace='pubchem', name=str(chemical))
        if chemical not in full_graph.nodes():
            continue
        chemicals_subgraph.append(chemical)
        for neighbor in full_graph.neighbors(chemical):
            chemicals_subgraph.append(neighbor)
    subgraphs_dict[cluster] = list(dict.fromkeys(chemicals_subgraph)) # to remove duplicates
fullgraph_edges = len(full_graph.edges())
cluster_weights = {}
for cluster, nodes in subgraphs_dict.items():
    subgraph = full_graph.subgraph(nodes)
    edges = len(subgraph.edges())
    cluster_weights[cluster] = edges/fullgraph_edges

cluster_df['weight'] = None
for cluster, weight in cluster_weights.items():
    cluster_df.loc[cluster_df['Cluster'] == cluster, 'weight'] == weight
cluster_df.to_csv('/home/bio/groupshare/rana/results/clusters_weights.csv')
