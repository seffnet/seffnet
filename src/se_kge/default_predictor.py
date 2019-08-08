import os

from .find_relations import Predictor

resources_path = os.path.join(os.pardir, "resources")

embeddings_path = os.path.abspath(
    os.path.join(resources_path, "predictive_model", "070819_node2vec_embeddings_complete01.embeddings"))
model_path = os.path.abspath(os.path.join(resources_path, "predictive_model", "070819_node2vec_model_complete01.pkl"))
graph_path = os.path.abspath(os.path.join(resources_path, "chemsim_50_graphs", "fullgraph_with_chemsim_50.edgelist"))
mapping_path = os.path.abspath(os.path.join(resources_path, "mapping", "fullgraph_nodes_mapping.tsv"))

predictor = Predictor.from_paths(
    model_path=model_path,
    embeddings_path=embeddings_path,
    graph_path=graph_path,
    mapping_path=mapping_path,
)
