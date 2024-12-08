from pathlib import Path
import json

def get_db_config():
    config_path = Path("data/config/db_config.json")
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    return cfg
