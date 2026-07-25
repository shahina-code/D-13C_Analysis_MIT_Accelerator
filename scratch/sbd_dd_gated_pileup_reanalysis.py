"""
SBD DD-Only Energy-Gated Pileup Re-Analysis
=============================================
Task from Ayushman/Shahina (July 25 meeting):
  Re-run ROOT files restricting to only the DD energy window [2.25-2.65 MeV].
  The previous pileup estimate used the ENTIRE spectrum — inflating the rate.
  We want the pure D+D reference proton rate to compute pileup correctly.
"""
import os, re, sys
import numpy as np

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("WARNING: uproot not installed. pip install uproot awkward")

# Energy windows (MeV)
DD_GATE       = (2.25, 2.65)   # Pure D+D proton reference peak
PILEUP_WINDOW = (4.60, 5.05)   # Sum-peak of two D+D protons ~4.84 MeV
SIGNAL_13Cd   = (5.08, 5.38)   # Real 13C+d proton line at 5.24 MeV
SIDEBAND      = (5.60, 6.60)   # Background sideband

TAU_HOLDOFF_s = 2500e-9   # 2500 ns hardware holdoff

base_dir = os.path.join(".", "13CD_cross_section_data-20260707T150327Z-3-001",
                        "13CD_cross_section_data", "13CD_cross_section_20260413", "DAQ")
if not os.path.exists(base_dir):
    base_dir = os.path.join("..", "13CD_cross_section_data-20260707T150327Z-3-001",
                            "13CD_cross_section_data", "13CD_cross_section_20260413", "DAQ")

if not os.path.exists(base_dir):
    print("ERROR: DAQ directory not found. Run from MAIN project directory.")
    raise SystemExit(1)

folders = sorted([f for f in os.listdir(base_dir)
                  if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))])
print(f"Found {len(folders)} run folders.")


def pileup_fraction(rate_hz, tau_s):
    return 1.0 - np.exp(-rate_hz * tau_s)


def extract_live_time(info_txt, ch):
    m = re.search(rf"{ch}@.*?(?=CH\d@|\Z)", info_txt, re.S)
    if not m:
        return None
    blk = m.group(0)
    lt = re.search(r"Live time\s*=\s*(\d+):(\d+):([\d.]+)", blk)
    if not lt:
        return None
    return int(lt.group(1)) * 3600 + int(lt.group(2)) * 60 + float(lt.group(3))


ch1_rows, ch2_rows = [], []
for fld in folders:
    info_p = os.path.join(base_dir, fld, f"{fld}_info.txt")
    if not os.path.exists(info_p):
        continue
    info_txt = open(info_p, encoding="utf-8", errors="replace").read()
    for ch, row_list in [("CH1", ch1_rows), ("CH2", ch2_rows)]:
        live = extract_live_time(info_txt, ch)
        if live is None or live <= 0:
            continue
        if not HAS_UPROOT:
            row_list.append({"run": fld, "live": live,
                             "n_dd": None, "n_pile": None,
                             "n_sig": None, "n_side": None})
            continue
        root_p = os.path.join(base_dir, fld, "FILTERED",
                              f"DataF_{ch}@N6724B_214_{fld}.root")
        if not os.path.exists(root_p):
            continue
        try:
            with uproot.open(root_p) as fh:
                if "Data_F" not in fh:
                    continue
                a = fh["Data_F"]["CalibEnergy"].array(library="np")
                n_dd   = int(((a >= DD_GATE[0])       & (a < DD_GATE[1])).sum())
                n_pile = int(((a >= PILEUP_WINDOW[0]) & (a < PILEUP_WINDOW[1])).sum())
                n_sig  = int(((a >= SIGNAL_13Cd[0])   & (a < SIGNAL_13Cd[1])).sum())
                n_side = int(((a >= SIDEBAND[0])       & (a < SIDEBAND[1])).sum())
                row_list.append({"run": fld, "live": live,
                                 "n_dd": n_dd, "n_pile": n_pile,
                                 "n_sig": n_sig, "n_side": n_side,
                                 "rate_dd": n_dd / live})
        except Exception as e:
            print(f"  WARNING {root_p}: {e}")


def summarize(rows, label):
    if not rows or rows[0].get("n_dd") is None:
        print(f"\n{label}: uproot unavailable.")
        return
    valid = [r for r in rows if r["n_dd"] is not None]
    if not valid:
        print(f"\n{label}: No valid ROOT files found.")
        return

    tot_dd   = sum(r["n_dd"]   for r in valid)
    tot_live = sum(r["live"]   for r in valid)
    tot_pile = sum(r["n_pile"] for r in valid)
    tot_sig  = sum(r["n_sig"]  for r in valid)
    tot_side = sum(r["n_side"] for r in valid)
    mean_rate = tot_dd / tot_live if tot_live > 0 else 0

    bkg_under_pile = (tot_side / (SIDEBAND[1]-SIDEBAND[0])) * (PILEUP_WINDOW[1]-PILEUP_WINDOW[0])
    net_pile = tot_pile - bkg_under_pile
    exp_pile = sum(r["n_dd"] * pileup_fraction(r["rate_dd"], TAU_HOLDOFF_s) for r in valid)
    tau_eff  = (net_pile / tot_dd) / mean_rate if (tot_dd > 0 and mean_rate > 0) else 0

    bkg_under_sig = (tot_side / (SIDEBAND[1]-SIDEBAND[0])) * (SIGNAL_13Cd[1]-SIGNAL_13Cd[0])
    net_sig = tot_sig - bkg_under_sig
    net_sig_err = np.sqrt(tot_sig + bkg_under_sig)

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {label} -- DD-ONLY GATED PILEUP ANALYSIS")
    print(f"{sep}")
    print(f"  Runs analyzed                  : {len(valid)}")
    print(f"  Total live time                : {tot_live:.1f} s ({tot_live/3600:.2f} h)")
    print(f"  DD-gate counts [2.25-2.65 MeV]: {tot_dd:,d}")
    print(f"  DD-only count rate (R_DD)      : {mean_rate:.4f} Hz")
    print()
    print(f"  --- Pileup Window [4.60-5.05 MeV] ---")
    print(f"  Observed counts in pile-up win : {tot_pile:,d}")
    print(f"  Sideband BG subtracted         : {bkg_under_pile:.1f}")
    print(f"  Net pile-up counts             : {net_pile:.1f}")
    print(f"  Expected pile-up (tau=2500 ns) : {exp_pile:.1f}")
    print(f"  Derived effective tau          : {tau_eff*1e9:.1f} ns  ({tau_eff*1e6:.3f} us)")
    print(f"  Hardware holdoff               : 2500.0 ns (2.500 us)")
    frac_diff = abs(tau_eff - TAU_HOLDOFF_s) / TAU_HOLDOFF_s * 100 if tau_eff > 0 else 0
    print(f"  Difference from hardware tau   : {frac_diff:.2f}%")
    print()
    print(f"  --- Signal Window [5.08-5.38 MeV] (13C+d line) ---")
    print(f"  Observed counts in signal win  : {tot_sig:,d}")
    print(f"  Sideband BG subtracted         : {bkg_under_sig:.1f}")
    print(f"  Net 13C+d signal counts        : {net_sig:.1f} +/- {net_sig_err:.1f}")
    print(f"{sep}")
    return dict(label=label, tot_dd=tot_dd, tot_live=tot_live, mean_rate=mean_rate,
                net_pile=net_pile, exp_pile=exp_pile, tau_eff=tau_eff,
                net_sig=net_sig, net_sig_err=net_sig_err)


r1 = summarize(ch1_rows, "CHANNEL 1 (CH1 - Primary SBD)")
r2 = summarize(ch2_rows, "CHANNEL 2 (CH2 - Reference Monitor)")

# Cross-check: does CH2 pileup match within 5%?
if r2 and r2['tau_eff'] > 0:
    pct = abs(r2['tau_eff'] - TAU_HOLDOFF_s) / TAU_HOLDOFF_s * 100
    print(f"\nCROSS-CHECK: CH2 tau_eff vs hardware holdoff: {pct:.2f}% difference")
    if pct < 5:
        print("CONFIRMED: 4.84 MeV peak on CH2 is DD pulse pileup (within 5% of hardware holdoff).")
    else:
        print("WARNING: tau_eff does not match hardware holdoff -- check energy windows.")
