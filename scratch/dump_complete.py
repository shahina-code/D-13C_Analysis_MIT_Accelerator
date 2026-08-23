import json

with open('MAIN/extra python/CR39_Analysis_Complete.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/complete_cells_dump.txt', 'w', encoding='utf-8') as f:
    for i, c in enumerate(nb['cells']):
        f.write(f"=== CELL {i:02d} [{c['cell_type']}] ===\n")
        f.write(''.join(c['source']))
        f.write("\n\n")

print(f"Dumped {len(nb['cells'])} cells to scratch/complete_cells_dump.txt")
