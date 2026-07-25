import os
import re
import uproot
import numpy as np

base_dir = r"c:\Users\sayak\Downloads\coding shits\physics\MAIN\13CD_cross_section_data-20260707T150327Z-3-001\13CD_cross_section_data\13CD_cross_section_20260413\DAQ"
folders = [f for f in os.listdir(base_dir) if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))]

LOW = (2.25, 2.65)
PILEUP_W = (4.60, 5.05)
SIGNAL_W = (5.08, 5.38)
SIDE = (5.60, 6.60)

ch1_recs = []

for fld in folders:
    info_p = os.path.join(base_dir, fld, f"{fld}_info.txt")
    root_p = os.path.join(base_dir, fld, "FILTERED", f"DataF_CH1@N6724B_214_{fld}.root")
    
    if not (os.path.exists(info_p) and os.path.exists(root_p)):
        continue
        
    txt = open(info_p, encoding="utf-8", errors="replace").read()
    m1 = re.search(r"CH1@.*?(?=CH2@|CH3@|\Z)", txt, re.S)
    if not m1:
        continue
    blk1 = m1.group(0)
    
    lt1 = re.search(r"Live time\s*=\s*(\d+):(\d+):([\d.]+)", blk1)
    if not lt1:
        continue
    live = int(lt1.group(1))*3600 + int(lt1.group(2))*60 + float(lt1.group(3))
    if live <= 0:
        continue
        
    with uproot.open(root_p) as fh:
        if "Data_F" in fh:
            a = fh["Data_F"]["CalibEnergy"].array(library="np")
            n_low = int(((a >= LOW[0]) & (a < LOW[1])).sum())
            n_pile = int(((a >= PILEUP_W[0]) & (a < PILEUP_W[1])).sum())
            n_sig = int(((a >= SIGNAL_W[0]) & (a < SIGNAL_W[1])).sum())
            n_side = int(((a >= SIDE[0]) & (a < SIDE[1])).sum())
            
            rate = n_low / live
            ch1_recs.append(dict(run=fld, live=live, n_low=n_low, n_pile=n_pile, n_sig=n_sig, n_side=n_side, rate=rate))

tot_low = sum(r["n_low"] for r in ch1_recs)
tot_pile_raw = sum(r["n_pile"] for r in ch1_recs)
tot_live = sum(r["live"] for r in ch1_recs)
tot_side = sum(r["n_side"] for r in ch1_recs)

bkg_density = tot_side / (SIDE[1] - SIDE[0])
bkg_in_pileup_win = bkg_density * (PILEUP_W[1] - PILEUP_W[0])
net_pileup_observed = tot_pile_raw - bkg_in_pileup_win

mean_rate = tot_low / tot_live

print("===============================================================")
print("  CHANNEL 1 (CH1) SBD ANALYSIS ACROSS ALL RUNS")
print("===============================================================")
print(f"Total valid runs       : {len(ch1_recs)}")
print(f"Total Live Time        : {tot_live:.1f} s ({tot_live/3600:.2f} h)")
print(f"Total D+D counts (N_low): {tot_low:,d}")
print(f"Mean count rate r      : {mean_rate:.2f} counts/s")
print(f"Raw pileup peak counts : {tot_pile_raw:,d}")
print(f"Sideband background    : {bkg_in_pileup_win:.1f}")
print(f"Net observed pileup    : {net_pileup_observed:.1f}")
print("---------------------------------------------------------------")

# Calculate Poisson Pile-Up for CH1 using hardware holdoff tau
# Tau calculation:
tau_exact = (net_pileup_observed / tot_low) / mean_rate

print(f"Exact resolving time tau to match observed net pileup (100% match):")
print(f"  tau = {tau_exact*1e9:.2f} ns  ({tau_exact*1e6:.4f} us)")
print()

# Check for tau = 2.5 us (2500 ns holdoff):
t_holdoff = 2500e-9
exp_pileup_per_run = sum(r["n_low"] * (1 - np.exp(-r["rate"] * t_holdoff)) for r in ch1_recs)
exp_pileup_global = tot_low * (1 - np.exp(-mean_rate * t_holdoff))

print(f"At Hardware Trigger Holdoff (t = 2.50 us = 2500 ns):")
print(f"  Per-run Poisson expected pileup : {exp_pileup_per_run:.1f} counts")
print(f"  Global rate Poisson expected    : {exp_pileup_global:.1f} counts")
print(f"  Observed net pileup counts      : {net_pileup_observed:.1f} counts")
diff_pct = abs(exp_pileup_per_run - net_pileup_observed) / net_pileup_observed * 100
print(f"  Difference: {diff_pct:.2f}%")
print("===============================================================")
