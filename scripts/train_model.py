# scripts/train_model.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import logging
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

logging.basicConfig(filename='data/logs/train_model.log', level=logging.INFO, format='%(asctime)s %(message)s')

class PatientDataset(Dataset):
    def __init__(self, path, pheno_classes, gene_classes):
        self.data = pd.read_json(path, lines=True)
        self.pheno_classes = pheno_classes
        self.gene_classes = gene_classes
        self.pheno_mlb = MultiLabelBinarizer(classes=pheno_classes)
        self.gene_mlb = MultiLabelBinarizer(classes=gene_classes)

        # Fit-transform phenotypes and genes
        self.data['pheno_vec'] = list(self.pheno_mlb.fit_transform(self.data['positive_phenotypes']))
        self.data['gene_vec'] = list(self.gene_mlb.fit_transform(self.data['all_candidate_genes']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pheno_vec = torch.tensor(row['pheno_vec'], dtype=torch.float32)
        gene_vec = torch.tensor(row['gene_vec'], dtype=torch.float32)
        clinical_vec = torch.tensor(row['clinical_vec'], dtype=torch.float32)
        x = torch.cat([pheno_vec, gene_vec, clinical_vec])
        y = torch.tensor([1.0 if row['true_diseases'] != 0 else 0.0], dtype=torch.float32)
        return x, y

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sig(self.fc2(x))
        return x

def collect_all_classes(paths):
    phenos = set()
    genes = set()
    for p in paths:
        df = pd.read_json(p, lines=True)
        for ph in df['positive_phenotypes']:
            phenos.update(ph)
        for gset in df['all_candidate_genes']:
            genes.update(gset)
    return sorted(list(phenos)), sorted(list(genes))

def run_simple_mlp_training():
    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/val.jsonl"
    test_path = "data/processed/test.jsonl"

    pheno_classes, gene_classes = collect_all_classes([train_path, val_path, test_path])

    train_ds = PatientDataset(train_path, pheno_classes, gene_classes)
    val_ds = PatientDataset(val_path, pheno_classes, gene_classes)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Determine input dimension
    sample_record = pd.read_json(train_path, lines=True).iloc[0]
    pheno_len = len(pheno_classes)
    gene_len = len(gene_classes)
    clinical_len = len(sample_record['clinical_vec'])
    input_dim = pheno_len + gene_len + clinical_len

    model = SimpleMLP(input_dim=input_dim, hidden_dim=128)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        preds_list = []
        true_list = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                predv = model(Xv)
                vloss = criterion(predv, yv)
                val_loss += vloss.item()
                predicted = (predv >= 0.5).float().flatten().tolist()
                truev = yv.flatten().tolist()
                preds_list.extend(predicted)
                true_list.extend(truev)
        avg_val_loss = val_loss / len(val_loader)
        precision = precision_score(true_list, preds_list, zero_division=0)
        recall = recall_score(true_list, preds_list, zero_division=0)
        f1 = f1_score(true_list, preds_list, zero_division=0)
        logging.info(
            f"Epoch {epoch+1}/{epochs}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}, Val Prec {precision:.4f}, Val Recall {recall:.4f}, Val F1 {f1:.4f}")

    os.makedirs("data/model", exist_ok=True)
    torch.save(model.state_dict(), "data/model/neuro_model.pt")
    logging.info("SimpleMLP training complete and saved.")

def run_random_forest_training():
    logging.info("Starting Random Forest training.")

    train_path = "data/processed/train.jsonl"
    val_path = "data/processed/val.jsonl"
    test_path = "data/processed/test.jsonl"

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    combined_df = pd.concat([train_df, val_df], axis=0)

    # Prepare phenotype/gene classes and encoders
    all_phenos = set()
    all_genes = set()
    for ph_list in combined_df['positive_phenotypes']:
        all_phenos.update(ph_list)
    for gene_list in combined_df['all_candidate_genes']:
        all_genes.update(gene_list)
    pheno_classes = sorted(list(all_phenos))
    gene_classes = sorted(list(all_genes))

    pheno_mlb = MultiLabelBinarizer(classes=pheno_classes)
    gene_mlb = MultiLabelBinarizer(classes=gene_classes)

    pheno_matrix = pheno_mlb.fit_transform(combined_df['positive_phenotypes'])
    gene_matrix = gene_mlb.fit_transform(combined_df['all_candidate_genes'])
    clinical_matrix = np.array(list(combined_df['clinical_vec']))
    X_combined = np.hstack([pheno_matrix, gene_matrix, clinical_matrix])
    y_combined = [1 if d != 0 else 0 for d in combined_df['true_diseases']]

    pheno_matrix_test = pheno_mlb.transform(test_df['positive_phenotypes'])
    gene_matrix_test = gene_mlb.transform(test_df['all_candidate_genes'])
    clinical_matrix_test = np.array(list(test_df['clinical_vec']))
    X_test = np.hstack([pheno_matrix_test, gene_matrix_test, clinical_matrix_test])
    y_test = [1 if d != 0 else 0 for d in test_df['true_diseases']]

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in skf.split(X_combined, y_combined):
        X_tr, X_val = X_combined[train_idx], X_combined[val_idx]
        y_tr, y_val = np.array(y_combined)[train_idx], np.array(y_combined)[val_idx]
        clf.fit(X_tr, y_tr)
        y_val_pred = clf.predict(X_val)
        fold_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
        f1_scores.append(fold_f1)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    logging.info(f"Cross-validated F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")

    clf.fit(X_combined, y_combined)
    y_test_pred = clf.predict(X_test)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    logging.info(
        f"Test results - Precision(macro): {test_precision:.4f}, Recall(macro): {test_recall:.4f}, F1(macro): {test_f1:.4f}")

    importances = clf.feature_importances_
    feature_names = pheno_classes + gene_classes + [f"clinical_{i}" for i in range(clinical_matrix.shape[1])]
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
    logging.info("Top 10 Feature Importances:\n" + importance_df.head(10).to_string(index=False))
    logging.info("Random Forest training complete.")

def main():
    run_simple_mlp_training()
    run_random_forest_training()

if __name__ == "__main__":
    main()
