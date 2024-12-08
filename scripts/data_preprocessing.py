# scripts/data_preprocessing.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pandas as pd
import logging
from utils.mappings import phenotype_to_hpo, gene_to_ensembl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Ensure logs directory exists
os.makedirs("data/logs", exist_ok=True)

# Configure logging
logging.basicConfig(filename='data/logs/preprocess.log', level=logging.INFO, format='%(asctime)s %(message)s')

def preprocess_data():
    # Load raw CSV data
    raw_path = "data/raw/patients.csv"
    df_raw = pd.read_csv(raw_path)

    # Process each patient record
    processed_records = []
    for _, row in df_raw.iterrows():
        patient_id = row.get("patient_id", None)
        gene_symbols = row.get("gene_symbols", "")
        phenotypes = row.get("phenotypes", "")
        disease_label_orig = row.get("disease_label", 0)

        # Convert disease_label into binary (0 if label=0, else 1)
        if isinstance(disease_label_orig, str):
            try:
                disease_label_orig = int(disease_label_orig)
            except ValueError:
                disease_label_orig = 1 if disease_label_orig != '0' else 0
        disease_label = 1 if disease_label_orig != 0 else 0

        # Split phenotypes and map to HPO terms
        phenotype_list = [p.strip() for p in phenotypes.split(",") if p.strip()]
        hpo_terms = []
        for p in phenotype_list:
            mapped_terms = phenotype_to_hpo(p)
            if mapped_terms:
                hpo_terms.extend(mapped_terms)

        # Map gene symbols to Ensembl IDs
        gene_list = [g.strip() for g in gene_symbols.split(",") if g.strip()]
        candidate_genes = []
        for gene_symbol in gene_list:
            ens_id = gene_to_ensembl(gene_symbol)
            if ens_id:
                candidate_genes.append(ens_id)

        # Skip if no HPO terms or genes
        if not hpo_terms or not candidate_genes:
            continue

        # Extract additional features
        age_at_onset = row.get("age_at_onset", None)
        family_history = row.get("family_history", 0)
        mri_findings = row.get("mri_findings", "")
        csf_protein = row.get("csf_protein", None)
        medication_history = row.get("medication_history", "")
        severity_score = row.get("severity_score", 0)

        # Append processed record
        processed_records.append({
            "id": patient_id,
            "positive_phenotypes": list(set(hpo_terms)),
            "all_candidate_genes": list(set(candidate_genes)),
            "true_diseases": disease_label,
            "age_at_onset": age_at_onset,
            "family_history": family_history,
            "mri_findings": mri_findings,
            "csf_protein": csf_protein,
            "medication_history": medication_history,
            "severity_score": severity_score
        })

    if not processed_records:
        logging.info("No records were processed.")
        return

    df = pd.DataFrame(processed_records)

    # Ensure categorical fields are strings
    df["mri_findings"] = df["mri_findings"].astype(str)
    df["medication_history"] = df["medication_history"].astype(str)

    # Fill missing numeric values if any
    numeric_features = ["age_at_onset", "csf_protein", "severity_score"]
    for nf in numeric_features:
        if df[nf].isnull().any():
            df[nf] = df[nf].fillna(df[nf].mean())

    # Ensure family_history is integer and not null
    df["family_history"] = df["family_history"].fillna(0).astype(int)

    # One-hot encode categorical features
    cat_features = ["mri_findings", "medication_history"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cat_matrix = ohe.fit_transform(df[cat_features])

    # Scale numeric features
    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(df[numeric_features])

    # Combine numeric, family_history, and categorical into one vector per patient
    final_features = []
    for i in range(len(df)):
        num_part = num_matrix[i]                    # numeric features as float array
        fam_part = np.array([df["family_history"].iloc[i]], dtype=float)
        cat_part = cat_matrix[i]                    # already 1D float array
        combined_vec = np.concatenate([num_part, fam_part, cat_part])
        final_features.append(combined_vec.tolist())

    df["clinical_vec"] = final_features

    # Drop intermediate columns
    drop_cols = cat_features + numeric_features + ["family_history"]
    df = df.drop(columns=drop_cols)

    # Stratified splitting
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['true_diseases'])
    train, val = train_test_split(train, test_size=0.2, random_state=42, stratify=train['true_diseases'])

    # Ensure output directory
    os.makedirs("data/processed", exist_ok=True)

    # Save splits
    train.to_json("data/processed/train.jsonl", orient='records', lines=True)
    val.to_json("data/processed/val.jsonl", orient='records', lines=True)
    test.to_json("data/processed/test.jsonl", orient='records', lines=True)

    logging.info("Preprocessing complete. Train, Validation, and Test sets created.")

if __name__ == "__main__":
    preprocess_data()
