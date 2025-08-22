import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def train_with_regularization(csv_file):
    print("Training gene expression model with regularization...")
    data = pd.read_csv(csv_file, index_col=0)
    gene_names = data.index.tolist()
    experiment_names = data.columns.tolist()
    
    # Parse experiment names to create feature matrix
    def parse_experiment_name(exp_name):
        parts = exp_name.split('+')
        perturbed_genes = []
        for part in parts:
            if part.startswith('g') and part[1:5].isdigit():
                gene_id = part.split('.')[0]
                perturbed_genes.append(gene_id)
        return perturbed_genes
    
    X = np.zeros((len(experiment_names), len(gene_names)))
    for exp_idx, exp_name in enumerate(experiment_names):
        perturbed_genes = parse_experiment_name(exp_name)
        for gene in perturbed_genes:
            if gene in gene_names:
                gene_idx = gene_names.index(gene)
                X[exp_idx, gene_idx] = 1
    
    # Add interaction terms
    n_genes = X.shape[1]
    interaction_terms = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            interaction_terms.append(X[:, i] * X[:, j])
    X_interactions = np.column_stack(interaction_terms)
    X_full = np.hstack([X, X_interactions])
    
    # Target matrix
    Y = data.values.T
    
    # Pipeline with scaling and Ridge regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    # Hyperparameter tuning for alpha
    param_grid = {'ridge__alpha': np.logspace(-3, 3, 7)}
    model = GridSearchCV(pipeline, param_grid, cv=5)
    model.fit(X_full, Y)
    
    print(f"Best alpha: {model.best_params_['ridge__alpha']}")
    print("Training complete.")
    
    return model, gene_names, n_genes

def predict_with_regularization(model, gene_names, n_genes, genes_to_perturb):
    # Validate input genes
    valid_genes = [gene for gene in genes_to_perturb if gene in gene_names]
    if not valid_genes:
        raise ValueError("No valid genes provided for prediction.")
    
    # Create feature vector
    features = np.zeros(n_genes)
    for gene in valid_genes:
        gene_idx = gene_names.index(gene)
        features[gene_idx] = 1
    
    # Generate interaction terms
    interaction_terms = []
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            interaction_terms.append(features[i] * features[j])
    features_full = np.hstack([features, interaction_terms])
    
    # Predict
    predictions = model.predict(features_full.reshape(1, -1))[0]
    
    return {gene: pred for gene, pred in zip(gene_names, predictions)}

# Usage
def run_improved(gene1, gene2):
    model, gene_names, n_genes = train_with_regularization('data/train_set.csv')
    genes_to_perturb = [gene1, gene2]
    predictions = predict_with_regularization(model, gene_names, n_genes, genes_to_perturb)
    return predictions