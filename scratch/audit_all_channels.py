import os
import re
import uproot

base_dir = r"c:\Users\sayak\Downloads\coding shits\physics\MAIN\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"
folders = [f for f in os.listdir(base_dir) if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))]

counts_per_ch = {"CH0": 0, "CH1": 0, "CH2": 0, "CH3": 0}

for fld in folders:
    for ch in ["CH0", "CH1", "CH2", "CH3"]:
        root_p = os.path.join(base_dir, fld, "FILTERED", f"DataF_{ch}@N6724B_214_{fld}.root")
        if os.path.exists(root_p):
            try:
                with uproot.open(root_p) as fh:
                    if "Data_F" in fh:
                        a = fh["Data_F"]["CalibEnergy"].array(library="np")
                        counts_per_ch[ch] += len(a)
            except:
                pass

print("TOTAL FILTERED EVENTS RECORDED IN ROOT FILES PER CHANNEL:")
for ch, cnt in counts_per_ch.items():
    print(f"  {ch}: {cnt:,d} events")
