# Brain_GML
Graph Machine Learning Project for CS6208.

Graph neural networks (GNNs) have shown promise in brain disorder research, particularly for modeling the brain connectome using graph-based approaches. The brain's functional network can be represented as a graph, where each node corresponds to a specific brain region as defined by the ROI and the edges represent the BOLD connectivity strength between them. Several recent studies have attempted to extend GNNs for analyzing brain networks. In this CS6208 Graph Machine Learning Project, I take inspiration from the ongoing research on brain disorders to develop an approach for constructing brain network graphs from fMRI data. My objective is to evaluate the performance of various GNN benchmarks on the fMRI dataset obtained from the Autism Brain Imaging Data Exchange (ABIDE) dataset. Specifically, each subject's fMRI data is parcellated into 200 regions of interest (ROIs) to create graphs, and compare the performance of five GNN benchmarks, namely Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), GATv2, GraphSAGE, and Graph Isormorphism Networks (GIN).

To first download the fMRI data, please run
```
python fetch_data.py
```

Next, further processing should be done using
```
python process_data.py
```

Lastly, to train various models, run
```
python train.py
```
