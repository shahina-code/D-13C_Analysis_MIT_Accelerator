import json

def inspect_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"=== {path} ({len(nb['cells'])} cells) ===")
    for i, cell in enumerate(nb['cells']):
        src = cell.get('source', [])
        first_line = src[0].strip().encode('ascii', 'replace').decode('ascii') if src else "EMPTY"
        print(f"Cell {i:02d} [{cell['cell_type']}]: {first_line[:80]}")

inspect_notebook('CR-39_Python_code/CR39_Analysis_Optimized.ipynb')
