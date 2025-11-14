
from pathlib import Path
import json

def ensure_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    ensure_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
