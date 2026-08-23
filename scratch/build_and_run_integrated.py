import json
import os
import sys

# 1. Read ScanData class code
with open('scratch/scandata_cell.py', 'r', encoding='utf-8') as f:
    scandata_code = f.read()

# Remove 'print("ScanData class defined")' from end of scandata_cell.py if present
scandata_code = scandata_code.replace('print(\'ScanData class defined\')', '').strip()

# 2. Read CR39_Analysis_Complete.ipynb
with open('MAIN/extra python/CR39_Analysis_Complete.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 02: Update imports
imports_code = (
    "import os\n"
    "import struct\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as patches\n"
    "from matplotlib.colors import LogNorm\n"
    "from matplotlib.gridspec import GridSpec\n"
    "from tqdm import tqdm\n\n"
    "print('Imports OK')\n"
)
nb['cells'][2]['source'] = imports_code.splitlines(keepends=True)

# Cell 04: Update config CPSA_FILE path handling
config_code = nb['cells'][4]['source']
for i, line in enumerate(config_code):
    if line.startswith('CPSA_FILE ='):
        config_code[i] = (
            'CPSA_FILE = "../CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"\n'
            'if not os.path.exists(CPSA_FILE) and os.path.exists("CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"):\n'
            '    CPSA_FILE = "CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"\n'
        )

# Cell 06: Replace ScanData placeholder
nb['cells'][6]['source'] = scandata_code.splitlines(keepends=True)

# Save integrated notebook in CR-39_Python_code/
integrated_path = 'CR-39_Python_code/CR39_Analysis_Integrated.ipynb'
with open(integrated_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Saved integrated notebook to {integrated_path}")

# Also update MAIN/extra python/CR39_Analysis_Complete.ipynb
complete_path = 'MAIN/extra python/CR39_Analysis_Complete.ipynb'
with open(complete_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print(f"Updated {complete_path} with working ScanData class")

# 3. Create standalone python script scratch/cr39_pipeline.py
py_lines = []
py_lines.append("#!/usr/bin/env python3")
py_lines.append("# CR-39 Analysis — Integrated Pipeline Script")
py_lines.append("# Generated automatically from CR39_Analysis_Complete.ipynb & ScanData reader")
py_lines.append("\n")

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        py_lines.append(f"# --- Cell {i:02d} ---")
        src = "".join(cell['source'])
        # strip plt.show() so it runs cleanly non-interactively
        src = src.replace("plt.show()", "# plt.show()")
        py_lines.append(src)
        py_lines.append("\n")

script_path = 'scratch/cr39_pipeline.py'
with open(script_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(py_lines))
print(f"Created standalone script at {script_path}")
