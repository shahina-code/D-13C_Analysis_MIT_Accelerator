import json

with open('CR-39_Python_code/CR39_Analysis_Optimized.ipynb') as f:
    nb = json.load(f)

total = len(nb['cells'])
angle_cells = [i for i, c in enumerate(nb['cells'])
               if 'INCIDENT ANGLE' in ''.join(c.get('source', []))]
print(f"CR39 notebook total cells : {total}")
print(f"Angle cells at positions  : {angle_cells}")

with open('pileup_estimate.ipynb') as f:
    nb2 = json.load(f)
print(f"Pileup notebook cells     : {len(nb2['cells'])}")
