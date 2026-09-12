# Traffic Flow Prediction with Spatiotemporal Graph Neural Networks

This project was completed as the final project for the Machine Learning course at National Yang Ming Chiao Tung University (NYCU).

## Overview

Traffic flow prediction is an important task in intelligent transportation systems. Traditional machine learning models often struggle to capture both spatial dependencies between road intersections and temporal dynamics of traffic patterns.

This project investigates several deep learning approaches for traffic forecasting, progressing from conventional neural networks to graph-based spatiotemporal models.

## Models

### Model 1: Multilayer Perceptron (MLP)

A baseline model that uses historical traffic flow observations as input features.

- Captures temporal information only
- Ignores spatial relationships between intersections
- Serves as a performance benchmark

### Model 2: Graph Convolutional Network (GCN)

A graph-based model that incorporates the road network structure.

- Represents intersections as graph nodes
- Represents road connections as graph edges
- Captures spatial dependencies among neighboring intersections

### Model 3: GCN-LSTM

A spatiotemporal model that combines graph neural networks with recurrent neural networks.

- GCN extracts spatial features from the road network
- LSTM captures temporal traffic dynamics
- Integrates spatial and temporal information for traffic forecasting

## Methodology

1. Construct a graph representation of the traffic network.
2. Process node features using graph convolution.
3. Learn temporal dependencies from historical observations.
4. Predict future traffic flow values.
5. Compare model performance across different architectures.

## Project Structure

```text
code_file/
├── readme.md
├── traffic_model_1.py   # MLP baseline
├── traffic_model_2.py   # Graph Convolutional Network
└── traffic_model_3.py   # GCN-LSTM model
```

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Graph Neural Networks (GNN)
- Long Short-Term Memory Networks (LSTM)

## Learning Outcomes

Through this project, I gained practical experience in:

- Graph neural networks
- Spatiotemporal modeling
- Traffic network analysis
- Model evaluation and comparison

## Author

Tsung-Yung Lin

Department of Applied Mathematics  
National Yang Ming Chiao Tung University
