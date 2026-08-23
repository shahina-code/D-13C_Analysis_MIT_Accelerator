import json

with open('CR-39_Python_code/CR39_Analysis_Optimized.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell6_code = ''.join(nb['cells'][6]['source'])
print("Cell 6 code length:", len(cell6_code))
print("Contains ScanData:", 'class ScanData' in cell6_code)

with open('scratch/scandata_cell.py', 'w', encoding='utf-8') as f:
    f.write(cell6_code)
print("Saved cell 6 code to scratch/scandata_cell.py")
