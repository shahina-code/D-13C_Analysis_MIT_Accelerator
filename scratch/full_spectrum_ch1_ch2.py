import os
import uproot
import numpy as np

base_dir = r"c:\Users\sayak\Downloads\coding shits\physics\MAIN\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"
folders = [f for f in os.listdir(base_dir) if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))]

ch1_all = []
ch2_all = []

for fld in folders:
    p1 = os.path.join(base_dir, fld, "FILTERED", f"DataF_CH1@N6724B_214_{fld}.root")
    p2 = os.path.join(base_dir, fld, "FILTERED", f"DataF_CH2@N6724B_214_{fld}.root")
    
    if os.path.exists(p1):
        try:
            with uproot.open(p1) as fh:
                if "Data_F" in fh:
                    ch1_all.append(fh["Data_F"]["CalibEnergy"].array(library="np"))
        except:
            pass
            
    if os.path.exists(p2):
        try:
            with uproot.open(p2) as fh:
                if "Data_F" in fh:
                    ch2_all.append(fh["Data_F"]["CalibEnergy"].array(library="np"))
        except:
            pass

a1 = np.concatenate(ch1_all) if ch1_all else np.array([])
a2 = np.concatenate(ch2_all) if ch2_all else np.array([])

print("=== FULL DATASET ENERGY SPECTRUM SUMMARY ===")
print("CHANNEL 1 (CH1): Total =", len(a1))
if len(a1) > 0:
    print(f"  Min = {a1.min():.3f}, Max = {a1.max():.3f}, Mean = {a1.mean():.3f}")
    print(f"  [2.25, 2.65] MeV (D+D peak)        : {((a1>=2.25)&(a1<2.65)).sum():,d}")
    print(f"  [4.60, 5.05] MeV (Pileup structure): {((a1>=4.60)&(a1<5.05)).sum():,d}")
    print(f"  [5.08, 5.38] MeV (13C+d signal)    : {((a1>=5.08)&(a1<5.38)).sum():,d}")
    print(f"  [5.60, 6.60] MeV (Background)      : {((a1>=5.60)&(a1<6.60)).sum():,d}")

print("\nCHANNEL 2 (CH2): Total =", len(a2))
if len(a2) > 0:
    print(f"  Min = {a2.min():.3f}, Max = {a2.max():.3f}, Mean = {a2.mean():.3f}")
    print(f"  [2.25, 2.65] MeV (D+D peak)        : {((a2>=2.25)&(a2<2.65)).sum():,d}")
    print(f"  [4.60, 5.05] MeV (Pileup structure): {((a2>=4.60)&(a2<5.05)).sum():,d}")
    print(f"  [5.08, 5.38] MeV (13C+d signal)    : {((a2>=5.08)&(a2<5.38)).sum():,d}")
    print(f"  [5.60, 6.60] MeV (Background)      : {((a2>=5.60)&(a2<6.60)).sum():,d}")
