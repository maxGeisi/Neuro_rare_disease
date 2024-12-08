# scripts/fetch_data.py
import json
import pandas as pd
from neo4j import GraphDatabase
import logging
from utils.config import get_db_config
from pathlib import Path
import os

# Setup logging
logging.basicConfig(filename='data/logs/fetch_data.log', level=logging.INFO)

def fetch_data(query_type=1):
    config = get_db_config()
    uri = config["uri"]
    username = config["username"]
    password = config["password"]
    database = config["database"]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    logging.info("Connected to Neo4j database.")

    if query_type == 0:
        query = """
        MATCH (b:Biological_sample)
        OPTIONAL MATCH (b)-[:HAS_PROTEIN]->(p:Protein)
        OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype)
        OPTIONAL MATCH (b)-[:HAS_DISEASE]->(d:Disease)
        RETURN b.subjectid AS subject_id, 
            collect(DISTINCT p.id) AS proteins,
            collect(DISTINCT ph.id) AS phenotypes,
            CASE WHEN d.name = 'control' THEN 0 ELSE 1 END AS disease
        """
    else:
        query = """
        MATCH (b:Biological_sample)-[:HAS_DISEASE]->(d:Disease)
        WHERE NOT d.name = 'control'
        OPTIONAL MATCH (b)-[:HAS_PROTEIN]->(p:Protein)
        OPTIONAL MATCH (b)-[:HAS_PHENOTYPE]->(ph:Phenotype)
        WITH b, 
            collect(DISTINCT p.id) AS proteins,
            collect(DISTINCT ph.id) AS phenotypes,
            d.synonyms AS synonyms
        UNWIND synonyms AS synonym
        WITH b, proteins, phenotypes, synonym
        WHERE synonym CONTAINS 'ICD10CM:'
        RETURN b.subjectid AS subject_id, 
            proteins,
            phenotypes,
            substring(synonym, size('ICD10CM:'), 1) AS disease
        """

    with driver.session(database=database) as session:
        result = session.run(query)
        data = []
        for record in result:
            data.append({
                "id": record['subject_id'],
                "pheno_type": record['phenotypes'] if 'phenotypes' in record else [],
                "protein": record['proteins'] if 'proteins' in record else [],
                "true_diseases": record['disease']
            })

    df = pd.DataFrame(data)
    os.makedirs("data/raw", exist_ok=True)
    df.to_json("data/raw/patients_raw.json", orient='records', indent=4)
    logging.info(f"Fetched {len(df)} records and saved to data/raw/patients_raw.json")

if __name__ == "__main__":
    fetch_data(query_type=1)
