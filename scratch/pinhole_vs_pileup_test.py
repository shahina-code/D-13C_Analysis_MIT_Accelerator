"""
Test the pinhole hypothesis against the pile-up hypothesis using the SBD data.

THE CLAIM: the 10 um Ta in front of the SBD has a small hole, so some protons
reach the detector without passing through the filter. That is said to explain
the 5.2 MeV peak, and to mean there is no pile-up.

THE TESTS:
  T1  A hole cannot be species-selective. If it passes 13C+d protons it must
      also pass D+D protons, which arrive UN-degraded at 3.02 MeV instead of
      2.457 MeV. Look for that satellite. Its size gives the hole's open-area
      fraction, which then PREDICTS the 13C+d through-hole yield.
  T2  A hole gives LESS material, so the through-hole peak must sit at HIGHER
      energy than the degraded peak. Check whether anything sits at 5.55 MeV.
  T3  A hole is geometric -> counts scale linearly with rate. Pile-up scales
      as rate squared. Re-run the rate-scaling test on each structure.
"""
import os, re, math
import numpy as np
import uproot

BASE = r"C:\Users\sayak\Downloads\coding shits\physics\MAIN"
DAQ = os.path.join(BASE, "13CD_cross_section_data-20260707T150327Z-3-001",
                   "13CD_cross_section_data", "13CD_cross_section_20260413", "DAQ")

folders = sorted(f for f in os.listdir(DAQ)
                 if f.startswith("2026") and os.path.isdir(os.path.join(DAQ, f)))


def blk(info_path, ch):
    txt = open(info_path, encoding="utf-8", errors="replace").read()
    m = re.search(rf"{ch}@.*?(?=CH\d@|\Z)", txt, re.S)
    return m.group(0) if m else ""


def tsec(block, key):
    m = re.search(rf"{key}\s*=\s*(\d+):(\d+):([\d.]+)", block)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def load(ch):
    arrs, recs = [], []
    for fld in folders:
        info_p = os.path.join(DAQ, fld, f"{fld}_info.txt")
        root_p = os.path.join(DAQ, fld, "FILTERED", f"DataF_{ch}@N6724B_214_{fld}.root")
        if not (os.path.exists(info_p) and os.path.exists(root_p)):
            continue
        b = blk(info_p, ch)
        live = tsec(b, "Live time")
        real = tsec(b, "Real time")
        if not live or live <= 0:
            continue
        with uproot.open(root_p) as fh:
            if "Data_F" not in fh:
                continue
            a = fh["Data_F"]["CalibEnergy"].array(library="np")
        arrs.append(a)
        recs.append(dict(run=fld, live=live, real=real or live, a=a))
    return np.concatenate(arrs), recs


def win(a, lo, hi):
    return int(((a >= lo) & (a < hi)).sum())


def netcount(a, sig, side):
    """Sideband-subtracted count in `sig`, background density from `side`."""
    raw = win(a, *sig)
    ns = win(a, *side)
    dens = ns / (side[1] - side[0])
    bkg = dens * (sig[1] - sig[0])
    net = raw - bkg
    err = math.sqrt(raw + bkg)
    return raw, bkg, net, err


for CH in ("CH2", "CH1"):
    print("#" * 78)
    print(f"#  {CH}")
    print("#" * 78)
    a, recs = load(CH)
    tlive = sum(r["live"] for r in recs)
    treal = sum(r["real"] for r in recs)
    print(f"runs={len(recs)}  events={len(a):,}  live={tlive/3600:.3f} h  real={treal/3600:.3f} h")

    # ---------------------------------------------------------------- T1
    # The D+D line and where an un-degraded 3.02 MeV satellite would be.
    print("\n-- T1: is there an UN-DEGRADED D+D satellite at 3.02 MeV? --")
    print("   (a hole that passes 13C+d protons MUST also pass D+D protons)")
    n_dd = win(a, 2.25, 2.65)
    # fine scan across the region between the degraded line and 3.02
    print("\n   fine scan, 0.05 MeV bins, 2.60 -> 3.40 MeV:")
    edges = np.arange(2.60, 3.45, 0.05)
    h, _ = np.histogram(a, bins=edges)
    for i in range(len(h)):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * min(60, int(h[i] / max(1, h.max()) * 60))
        mark = "   <-- 3.02 MeV un-degraded D+D would be HERE" if lo <= 3.02 < hi else ""
        print(f"     {lo:.2f}-{hi:.2f}  {h[i]:>8,d}  {bar}{mark}")

    raw302, bkg302, net302, err302 = netcount(a, (2.94, 3.10), (3.30, 3.80))
    print(f"\n   D+D degraded line (2.25-2.65)     : {n_dd:,}")
    print(f"   3.02 MeV window (2.94-3.10)  raw  : {raw302:,}")
    print(f"                                bkg  : {bkg302:.1f}")
    print(f"                                NET  : {net302:.1f} +/- {err302:.1f}")
    if net302 > 0 and n_dd > 0:
        frac = net302 / n_dd
        print(f"   implied OPEN-AREA FRACTION of hole : {frac:.3e}  ({frac*100:.4f}%)")
    else:
        print("   implied OPEN-AREA FRACTION of hole : consistent with ZERO")

    # ---------------------------------------------------------------- T2
    print("\n-- T2: where does the extra structure sit relative to 5.2? --")
    print("   pinhole predicts un-degraded 13C+d at 5.55 MeV (HIGHER, less material)")
    print("   pile-up predicts a sum peak at 2 x 2.457 = 4.915 MeV (LOWER)")
    for name, w in [("4.84 structure", (4.60, 5.05)),
                    ("5.24 structure", (5.08, 5.38)),
                    ("5.55 un-degraded", (5.42, 5.68))]:
        r_, b_, n_, e_ = netcount(a, w, (5.80, 6.80))
        sig = n_ / e_ if e_ > 0 else 0
        print(f"   {name:<18} {w}  raw {r_:>5,d}  bkg {b_:>6.1f}  NET {n_:>7.1f} +/- {e_:.1f}  ({sig:+.1f} sigma)")

    # ---------------------------------------------------------------- T3
    print("\n-- T3: rate scaling. hole/real => FLAT.  pile-up => RISES. --")
    for r in recs:
        r["n_low"] = win(r["a"], 2.25, 2.65)
        r["rate"] = r["n_low"] / r["live"]
        r["n48"] = win(r["a"], 4.60, 5.05)
        r["n52"] = win(r["a"], 5.08, 5.38)
        r["n_side"] = win(r["a"], 5.80, 6.80)
        r["nsum"] = r["n48"] + r["n52"]
    recs2 = sorted(recs, key=lambda r: r["rate"])
    sd = sum(r["n_side"] for r in recs2) / 1.0  # counts per MeV over 1.0 MeV
    NG = 4
    cum = np.cumsum([r["n_low"] for r in recs2])
    idx = [0] + [int(np.searchsorted(cum, cum[-1] * k / NG)) + 1 for k in range(1, NG)] + [len(recs2)]

    def scaling(key, bw, label):
        rs, ys, es = [], [], []
        for k in range(NG):
            sub = recs2[idx[k]:idx[k + 1]]
            if not sub:
                continue
            nl = sum(r["n_low"] for r in sub)
            nraw = sum(r[key] for r in sub)
            bkg = sd * bw * (sum(r["live"] for r in sub) / tlive)
            mr = sum(r["rate"] * r["live"] for r in sub) / sum(r["live"] for r in sub)
            ratio = (nraw - bkg) / nl
            err = abs(ratio) * math.sqrt(1 / max(nraw, 1) + 1 / nl)
            rs.append(mr); ys.append(ratio); es.append(err)
        r_, y_, e_ = map(np.array, (rs, ys, es))
        if len(r_) < 3:
            print(f"   {label}: too few groups"); return
        w = 1 / e_ ** 2
        const = np.sum(w * y_) / np.sum(w)
        chi2c = np.sum(((y_ - const) / e_) ** 2)
        A = np.vstack([np.ones_like(r_), r_]).T
        cov = np.linalg.inv(A.T @ np.diag(w) @ A)
        beta = cov @ (A.T @ np.diag(w) @ y_)
        chi2l = np.sum(((y_ - A @ beta) / e_) ** 2)
        sig = beta[1] / np.sqrt(cov[1, 1])
        verdict = "RISES WITH RATE -> pile-up" if abs(sig) >= 2 else "FLAT -> real/geometric"
        print(f"   {label}")
        print(f"     rates      : {np.round(r_,1)}")
        print(f"     ratio x1e4 : {np.round(y_*1e4,3)}")
        print(f"     flat model chi2={chi2c:6.2f}/{len(r_)-1}   linear chi2={chi2l:6.2f}/{len(r_)-2}")
        print(f"     slope = {sig:+.1f} sigma   => {verdict}")

    scaling("n48", 0.45, "4.84 MeV structure alone")
    scaling("n52", 0.30, "5.24 MeV structure alone")
    scaling("nsum", 0.75, "4.84 + 5.24 SUMMED  (as requested)")
    print()
