this project aimed at enhancing the prediction of protein–ligand binding affinity, a critical aspect of drug discovery. By integrating multiple Graph Neural Network architectures—Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Graph Isomorphism Networks (GIN), and GraphSAGE—within the GraphLambda framework, this model captures diverse molecular interactions more effectively than single-model approaches.

Key features of GNNFusion include:

Multi-GNN Integration: Combines the strengths of various GNN architectures to improve predictive accuracy.

Explainable AI: Utilizes GNNExplainer to provide insights into the model's decision-making process, highlighting important atoms and bonds influencing predictions.

Robust Evaluation: Assessed on the CASF-2016 benchmark dataset across multiple challenging splits, achieving an RMSE of 0.76 kcal/mol and a Pearson correlation coefficient of 0.94 in the most stringent complex–complex split.

Reproducibility: Includes comprehensive documentation and scripts to facilitate replication and further exploration by researchers.
