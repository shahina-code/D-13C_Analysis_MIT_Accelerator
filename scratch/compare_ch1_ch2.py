import os
import re
import uproot
import numpy as np

base_dir = r"c:\Users\sayak\Downloads\coding shits\physics\MAIN\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"
folders = [f for f in os.listdir(base_dir) if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))]

ch1_energies = []
ch2_energies = []

for fld in folders[:10]: # Check first 10 runs
    p1 = os.path.join(base_dir, fld, "FILTERED", f"DataF_CH1@N6724B_214_{fld}.root")
    p2 = os.path.join(base_dir, fld, "FILTERED", f"DataF_CH2@N6724B_214_{fld}.root")
    
    if os.path.exists(p1):
        with uproot.open(p1) as fh:
            if "Data_F" in fh:
                a1 = fh["Data_F"]["CalibEnergy"].array(library="np")
                ch1_energies.append(a1)
                
    if os.path.exists(p2):
        with uproot.open(p2) as fh:
            if "Data_F" in fh:
                a2 = fh["Data_F"]["CalibEnergy"].array(library="np")
                ch2_energies.append(a2)

if ch1_energies:
    all_c1 = np.concatenate(ch1_energies)
    print("CH1 Filtered spectrum (10 runs): total =", len(all_c1))
    print("  Min energy:", all_c1.min(), "Max energy:", all_c1.max())
    print("  Counts in [2.25, 2.65] (D+D):", ((all_c1 >= 2.25) & (all_c1 < 2.65)).sum())
    print("  Counts in [4.60, 5.05] (Pileup peak):", ((all_c1 >= 4.60) & (all_c1 < 5.05)).sum())
    print("  Counts in [5.08, 5.38] (Real signal):", ((all_c1 >= 5.08) & (all_c1 < 5.38)).sum())

if ch2_energies:
    all_c2 = np.concatenate(ch2_energies)
    print("\nCH2 Filtered spectrum (10 runs): total =", len(all_c2))
    print("  Min energy:", all_c2.min(), "Max energy:", all_c2.max())
    print("  Counts in [2.25, 2.65] (D+D):", ((all_c2 >= 2.25) & (all_c2 < 2.65)).sum())
    print("  Counts in [4.60, 5.05] (Pileup peak):", ((all_c2 >= 4.60) & (all_c2 < 5.05)).sum())
    print("  Counts in [5.08, 5.38] (Real signal):", ((all_c2 >= 5.08) & (all_c2 < 5.38)).sum())
