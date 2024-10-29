**Blood-Brain Barrier Prediction using Machine Learning and Graph Neural Networks**

This repository demonstrates my expertise in applying machine learning models to predict blood-brain barrier (BBB) permeability, with a focus on using RDKit for molecular fingerprinting and property extraction, and developing a custom Graph Neural Network (GNN) model for enhanced classification. The project allowed me to integrate both traditional machine learning and cutting-edge deep learning methods, showcasing a comprehensive skill set in machine learning for drug discovery and molecular analysis.

**Project Overview**

In this project, I used SMILES data to construct a dataset containing RDKit-based molecular descriptors, which were then utilized as features for classical machine learning models and a custom-designed GNN. The objective was to predict BBB permeability, labeling compounds as "Active" or "Inactive" based on their molecular structure.

**Feature Engineering: Molecular Descriptors**

Using RDKit, I generated two types of descriptors that served as input features for my models:

1. **RDKit Morgan Molecular Fingerprints as Descriptors**: Extended-connectivity fingerprints were created to represent structural and chemical properties of molecules in feature vectors, enhancing the model's ability to capture molecular patterns.
2. **RDKit Molecular Properties as Descriptors**: I computed key molecular properties, such as molecular weight, logP, topological polar surface area (TPSA), atom count, and bond count, which provide meaningful insights into molecular behavior and interactions relevant to BBB permeability.

**Classical Machine Learning Models**

To establish a robust baseline, I employed the following algorithms for classification using the molecular descriptors as features:

- **AdaBoost Classifier**
- **Decision Tree Classifier**
- **Extra Trees Classifier**
- **Gradient Boosting Classifier**
- **K-Neighbors Classifier**
- **Random Forest Classifier**
- **Ridge Classifier**
- **XGBoost Classifier**

Each of these models provided insights into how traditional machine learning approaches perform on RDKit-generated descriptors for BBB permeability prediction.

**Graph Neural Network (GNN)**

The GNN served as the central model in this project, built with PyTorch Geometric, and tailored for molecular data analysis. Using SMILES data, I converted each molecule into a graph structure to capture atomic interactions and bonding configurations, enhancing the predictive capacity beyond traditional descriptors.

**GNN Methodology**:

- **Node Features**: Atoms were represented as nodes, with attributes derived from atom types, hybridization, aromaticity, and charge.
- **Edge Connections**: Bonds between atoms were encoded as edges, facilitating the GNN's ability to learn both intra- and intermolecular relationships.
- **Training**: The GNN was trained with a 5-fold cross-validation approach, using advanced metrics to evaluate the classification performance.

This model effectively harnessed both the spatial and chemical structure of molecules, proving particularly effective in capturing complex molecular interactions that are less apparent with classical descriptors alone.

**Methodology**

**Data Preprocessing**

1. **Feature Engineering**: Generated Morgan fingerprints and molecular properties from RDKit, resulting in a comprehensive descriptor set.
2. **Graph Conversion**: SMILES strings were transformed into molecular graphs for GNN input, encapsulating detailed structural data.

**Model Evaluation**

Each model was rigorously evaluated using cross-validation, with metrics such as:

- **Accuracy**: Classification accuracy across folds.
- **F1 Score**: Evaluation of the model's precision and recall, important given the class imbalance.
- **ROC-AUC Score**: Reflects the model's overall ability to distinguish between "Active" and "Inactive" compounds.

**Training and Evaluation of GNN**

The GNN model was evaluated using:

- **Confusion Matrix**: Visualized true versus predicted labels to assess prediction accuracy.
- **Loss Curve**: Plotted training loss to monitor model stability and convergence.

**Key Takeaways**

This project allowed me to:

- Master RDKit for molecular descriptor generation, including both fingerprinting and property extraction.
- Build and optimize traditional and advanced machine learning models for molecular data.
- Design, train, and evaluate a Graph Neural Network that leverages molecular graph structures for improved prediction accuracy.

**Requirements**

- **Python Libraries**: pandas, seaborn, matplotlib, scikit-learn, rdkit, torch, torch_geometric, and xgboost.

**How to Run**

1. Install dependencies: pip install -r requirements.txt
2. Execute the script in a Jupyter notebook or Python environment.# classification
