import json
import os

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MAPPINGS_DIR = os.path.join(BASE_DIR, "utils", "mappings")

with open(os.path.join(MAPPINGS_DIR, "phenotype_to_hp.json"), "r") as f:
    PHENOTYPE_MAP = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "hpo_map.json"), 'r') as f:
    HPO_MAP = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "protein_to_gene.json"), 'r') as f:
    PROT2GENE = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "gene_symbol_to_ensembl.json"), 'r') as f:
    GENE2ENSEMBL = json.load(f)

def phenotype_to_hpo(pheno: str):
    # Check main phenotype map
    if pheno in PHENOTYPE_MAP:
        return PHENOTYPE_MAP[pheno]
    # Check fallback map
    if pheno in HPO_MAP:
        return [HPO_MAP[pheno]]
    # Return empty if no match
    return []


def protein_to_gene(prot_id: str):
    # Convert protein ID to gene symbol
    return PROT2GENE.get(prot_id, None)


def gene_to_ensembl(gene_symbol: str):
    # Convert gene symbol to Ensembl ID
    return GENE2ENSEMBL.get(gene_symbol, None)
