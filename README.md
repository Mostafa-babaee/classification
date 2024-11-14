**Blood-Brain Barrier Prediction using Machine Learning and Graph Neural Networks**

This repository demonstrates my expertise in predicting blood-brain barrier (BBB) permeability, focusing on RDKit for molecular fingerprinting and property extraction, alongside a custom Graph Neural Network (GNN) model for advanced classification. The project integrates traditional machine learning and state-of-the-art deep learning methods, showcasing a comprehensive skill set in machine learning for drug discovery and molecular analysis.

**Project Overview**

In this project, SMILES data was used to construct a dataset containing RDKit-based molecular descriptors, which were then utilized as features for classical machine learning models and a custom GNN. The goal was to predict BBB permeability, labeling compounds as "Active" or "Inactive" based on their molecular structure.

**Feature Engineering: Molecular Descriptors**

Using RDKit, I generated two types of descriptors that served as input features:

1. **RDKit Morgan Molecular Fingerprints**: Extended-connectivity fingerprints representing structural and chemical properties of molecules, enhancing the model's ability to capture molecular patterns.
2. **RDKit Molecular Properties**: Calculated key molecular properties, such as molecular weight, logP, TPSA, atom count, and bond count, which provide insights into BBB permeability.

**Classical Machine Learning Models**

To establish a robust baseline, I employed the following algorithms for classification using the molecular descriptors:

- AdaBoost Classifier
- Decision Tree Classifier
- Extra Trees Classifier
- Gradient Boosting Classifier
- K-Neighbors Classifier
- Random Forest Classifier
- Ridge Classifier
- XGBoost Classifier

These models provided insights into how traditional machine learning approaches perform on RDKit-generated descriptors for BBB permeability prediction.

**Graph Neural Network (GNN)**

The GNN model, implemented with PyTorch Geometric, was tailored for molecular data analysis. SMILES data was converted into a graph structure to capture atomic interactions and bonding configurations, enhancing predictive capacity beyond traditional descriptors.

**GNN Methodology:**

- **Node Features**: Atoms were represented as nodes, with attributes derived from atom types, hybridization, aromaticity, and charge.
- **Edge Connections**: Bonds between atoms were encoded as edges, facilitating the GNN's learning of both intra- and intermolecular relationships.
- **Training**: The GNN was trained with a 5-fold cross-validation approach, using advanced metrics to evaluate classification performance.

This model effectively captured complex molecular interactions, offering advantages over classical descriptors alone.

**Methodology**

**Data Preprocessing**

1. **Feature Engineering**: Generated Morgan fingerprints and molecular properties from RDKit, creating a comprehensive descriptor set.
2. **Graph Conversion**: SMILES strings were transformed into molecular graphs for GNN input, encapsulating detailed structural data.

**Model Evaluation**

Each model was evaluated using cross-validation with metrics such as:

- **Accuracy**: Classification accuracy across folds.
- **F1 Score**: Evaluation of precision and recall, crucial for class imbalance.
- **ROC-AUC Score**: Reflects the model's ability to distinguish between "Active" and "Inactive" compounds.

**GNN Training and Evaluation**

The GNN model was assessed with:

- **Confusion Matrix**: Visualized true versus predicted labels to assess accuracy.
- **Loss Curve**: Monitored training loss for stability and convergence.

**Key Takeaways**

This project allowed me to:

- Master RDKit for molecular descriptor generation, including fingerprinting and property extraction.
- Build and optimize traditional and advanced machine learning models for molecular data.
- Design, train, and evaluate a GNN model leveraging molecular graph structures for improved prediction accuracy.

**Results**

**Using RDKit Morgan Molecular Fingerprints as Descriptors:**

- **AdaBoost**: Accuracy = 0.851, F1 = 0.909, AUC = 0.709
- **Decision Tree**: Accuracy = 0.842, F1 = 0.896, AUC = 0.781
- **Extra Trees**: Accuracy = 0.889, F1 = 0.930, AUC = 0.802
- **Gradient Boosting**: Accuracy = 0.877, F1 = 0.924, AUC = 0.770
- **K-Neighbors**: Accuracy = 0.866, F1 = 0.918, AUC = 0.744
- **Random Forest**: Accuracy = 0.887, F1 = 0.929, AUC = 0.791
- **Ridge**: Accuracy = 0.854, F1 = 0.906, AUC = 0.787
- **XGBoost**: Accuracy = 0.884, F1 = 0.927, AUC = 0.804

**Using RDKit Molecular Properties as Descriptors:**

- **AdaBoost**: Accuracy = 0.839, F1 = 0.899, AUC = 0.727
- **Decision Tree**: Accuracy = 0.807, F1 = 0.874, AUC = 0.732
- **Extra Trees**: Accuracy = 0.847, F1 = 0.903, AUC = 0.756
- **Gradient Boosting**: Accuracy = 0.844, F1 = 0.903, AUC = 0.726
- **K-Neighbors**: Accuracy = 0.825, F1 = 0.890, AUC = 0.706
- **Random Forest**: Accuracy = 0.847, F1 = 0.903, AUC = 0.753
- **Ridge**: Accuracy = 0.832, F1 = 0.899, AUC = 0.672
- **XGBoost**: Accuracy = 0.842, F1 = 0.899, AUC = 0.754

**Graph Neural Network (GNN)**

- **Average Accuracy (5 folds)**: 0.815
- **Average AUC (5 folds)**: 0.782
