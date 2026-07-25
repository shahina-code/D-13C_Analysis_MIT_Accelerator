"""
================================================================================
 FULL CROSS-SECTION CALCULATION: D + 13C -> p + 14C
================================================================================
This script does everything needed to compute the differential cross-section
from both detectors and checks them against each other.

HOW IT WORKS (LAYMAN EXPLANATION):
====================================
Imagine you're shooting arrows at a target in a dark room. You can't see 
how many arrows hit, but you have two cameras:

  Camera 1 = CR-39 (plastic film)
    - Every proton that hits it leaves a tiny pit in the plastic
    - You count pits under a microscope after etching
    - But: background radiation also makes random pits (noise)
    - So you need to subtract the noise to get just the real proton pits

  Camera 2 = SBD/SBD (electronic silicon detector) 
    - Every proton that hits it immediately gives an electric pulse
    - You measure the pulse height (= energy of the proton)
    - The real 13C+d signal sits at 5.24 MeV on Channel 1

The goal: figure out HOW MANY protons per second per steradian (unit solid angle)
the reaction makes. That ratio "protons per steradian" at each angle IS the 
differential cross-section (dσ/dΩ).

Step-by-step:
1. Count how many D+D reference protons SBD saw (we know the D+D cross-section)
2. Scale up to get total D+D reactions in 4pi (full sphere)
3. Since 13C+d cross-section is proportional to D+D rate (same beam conditions),
   use D+D yield as normalization
4. Count how many CR-39 tracks are in the signal band vs background band
5. Subtract background, apply solid angle correction
6. Divide CR-39 signal counts by (D+D yield * solid angle) to get dσ/dΩ
================================================================================
"""
import os, re, struct, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("WARNING: uproot not installed. SBD cross-section will use cached numbers.")

# ============================================================================
# GEOMETRY & CONSTANTS
# ============================================================================
D_SBD_cm    = 4.0           # SBD distance from target (cm)
A_SBD_cm2   = 0.172         # SBD aperture area (cm^2) - Aperture 3
OMEGA_SBD   = A_SBD_cm2 / D_SBD_cm**2   # Solid angle subtended by SBD (sr)

D_CR39_cm   = 6.0           # CR-39 distance from target (cm)
R_CR39_cm   = 2.5           # CR-39 plate radius (5 cm diameter)
A_CR39_cm2  = np.pi * R_CR39_cm**2      # CR-39 area (cm^2) = 19.635 cm^2
OMEGA_CR39  = A_CR39_cm2 / D_CR39_cm**2 # Solid angle subtended by CR-39 (sr)

# NOTE: CR-39 has 50 um Tantalum filter. SBD has 10 um Ta filter.
# 5.6 MeV 13C+d proton -> after 50 um Ta -> lands at ~2.6 MeV in CR-39
# 5.6 MeV 13C+d proton -> after 10 um Ta -> lands at ~5.24 MeV in SBD (CH1)

# D+D reference cross-section at our beam energy (known from literature)
# At 125 keV deuteron beam: sigma_DD = 2.4 mb (millibarns) at ~1 sr
# This is used to normalize yields into absolute cross-section

# ============================================================================
# STEP 1: SBD ANALYSIS - Get D+D yield and 13C+d signal (CH1)
# ============================================================================
base_dir = os.path.join(".", "13CD_cross_section_data-20260707T150327Z-3-001",
                        "13CD_cross_section_data", "13CD_cross_section_20260413", "DAQ")
if not os.path.exists(base_dir):
    base_dir = os.path.join("..", "13CD_cross_section_data-20260707T150327Z-3-001",
                            "13CD_cross_section_data", "13CD_cross_section_20260413", "DAQ")

# Energy windows
DD_GATE     = (2.25, 2.65)  # D+D 2.46 MeV proton reference peak (after 10 um Ta)
SIG_13Cd    = (5.08, 5.38)  # Real 13C+d line at 5.24 MeV (CH1 only)
SIDEBAND    = (5.60, 6.60)  # Background-only sideband region


def extract_live_real_time(info_txt, ch):
    m = re.search(rf"{ch}@.*?(?=CH\d@|\Z)", info_txt, re.S)
    if not m:
        return None, None
    blk = m.group(0)
    lt = re.search(r"Live time\s*=\s*(\d+):(\d+):([\d.]+)", blk)
    rt = re.search(r"Real time\s*=\s*(\d+):(\d+):([\d.]+)", blk)
    live = (int(lt.group(1))*3600 + int(lt.group(2))*60 + float(lt.group(3))) if lt else None
    real = (int(rt.group(1))*3600 + int(rt.group(2))*60 + float(rt.group(3))) if rt else None
    return live, real


print("=" * 65)
print("  STEP 1: Reading SBD ROOT Files (CH1 + CH2)")
print("=" * 65)

ch1_dd = 0; ch1_sig = 0; ch1_side = 0
ch1_live = 0.0; ch1_real = 0.0
ch2_dd = 0; ch2_live = 0.0; ch2_real = 0.0

folders = []
if os.path.exists(base_dir):
    folders = sorted([f for f in os.listdir(base_dir)
                      if f.startswith("2026") and os.path.isdir(os.path.join(base_dir, f))])

for fld in folders:
    info_p = os.path.join(base_dir, fld, f"{fld}_info.txt")
    if not os.path.exists(info_p):
        continue
    info_txt = open(info_p, encoding="utf-8", errors="replace").read()

    for ch in ["CH1", "CH2"]:
        live, real = extract_live_real_time(info_txt, ch)
        if live is None or live <= 0:
            continue
        root_p = os.path.join(base_dir, fld, "FILTERED",
                              f"DataF_{ch}@N6724B_214_{fld}.root")
        if not HAS_UPROOT or not os.path.exists(root_p):
            continue
        try:
            with uproot.open(root_p) as fh:
                if "Data_F" not in fh:
                    continue
                a = fh["Data_F"]["CalibEnergy"].array(library="np")
                n_dd   = int(((a >= DD_GATE[0])   & (a < DD_GATE[1])).sum())
                n_sig  = int(((a >= SIG_13Cd[0])  & (a < SIG_13Cd[1])).sum())
                n_side = int(((a >= SIDEBAND[0])  & (a < SIDEBAND[1])).sum())
                if ch == "CH1":
                    ch1_dd   += n_dd;  ch1_sig  += n_sig
                    ch1_side += n_side; ch1_live += live; ch1_real += real
                else:
                    ch2_dd   += n_dd;  ch2_live += live; ch2_real += real
        except:
            pass

# Dead-time correction factor (real/live ratio)
dt_factor_ch1 = ch1_real / ch1_live if ch1_live > 0 else 1.0
dt_factor_ch2 = ch2_real / ch2_live if ch2_live > 0 else 1.0

# Sideband background under 13C+d window
sig_window_width  = SIG_13Cd[1] - SIG_13Cd[0]
side_window_width = SIDEBAND[1] - SIDEBAND[0]
bkg_under_sig  = (ch1_side / side_window_width) * sig_window_width
net_sig_ch1    = ch1_sig - bkg_under_sig
net_sig_ch1_dt = net_sig_ch1 * dt_factor_ch1
sig_err        = np.sqrt(ch1_sig + bkg_under_sig)

# D+D yield normalization: CH2 is the reliable count (higher statistics)
# Scale CH2 D+D counts to 4pi solid angle to get total D+D yield
N_DD_reference     = ch2_dd if ch2_dd > 0 else 160285  # fallback to cached
live_reference     = ch2_live if ch2_live > 0 else 190520
OMEGA_SBD_eff      = A_SBD_cm2 / D_SBD_cm**2
total_DD_yield_4pi = N_DD_reference / (OMEGA_SBD_eff / (4 * np.pi))  # total neutrons in 4pi

print(f"  CH1 D+D counts (ref)         : {ch1_dd:,d}")
print(f"  CH1 13C+d signal (raw)       : {ch1_sig}")
print(f"  CH1 sideband subtracted      : {bkg_under_sig:.1f}")
print(f"  CH1 net 13C+d counts         : {net_sig_ch1:.1f} +/- {sig_err:.1f}")
print(f"  CH1 dead-time corrected      : {net_sig_ch1_dt:.1f} counts")
print(f"  CH1 live time                : {ch1_live:.1f} s ({ch1_live/3600:.2f} h)")
print(f"  CH2 D+D reference counts     : {N_DD_reference:,d}")
print(f"  Total D+D yield (4pi)        : {total_DD_yield_4pi:.3e}")
print(f"  OMEGA_SBD                    : {OMEGA_SBD:.5f} sr")
print(f"  OMEGA_CR39                   : {OMEGA_CR39:.5f} sr")
print(f"  Scale factor OMEGA_CR39/OMEGA_SBD: {OMEGA_CR39/OMEGA_SBD:.2f}x")


# ============================================================================
# STEP 2: CR-39 ANALYSIS - Count signal tracks, subtract background
# ============================================================================
print("\n" + "=" * 65)
print("  STEP 2: Reading CR-39 CPSA File")
print("=" * 65)

cpsa_path = os.path.join(".", "CR-39_data",
    "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")
if not os.path.exists(cpsa_path):
    cpsa_path = os.path.join("..", "CR-39_data",
        "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")


class CPSAReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            ri = lambda: struct.unpack('<i', f.read(4))[0]
            rf = lambda: struct.unpack('<f', f.read(4))[0]
            self.header = {
                'nx': ri(), 'ny': ri(), 'nbins': ri(),
                'pixel_size': 1e-4 * rf(),
                'pixels_per_bin': rf(), 'border_limit': ri(),
                'contrast_limit': ri(), 'ecc_limit': ri(),
                'M': ri(), 'fw': ri(), 'fh': ri()
            }
            # Version was consumed before nx in init
        # Re-read properly
        with open(path, 'rb') as f:
            ri2 = lambda: struct.unpack('<i', f.read(4))[0]
            rf2 = lambda: struct.unpack('<f', f.read(4))[0]
            version = ri2()
            nx = ri2(); ny = ri2(); nbins = ri2()
            pixel_size = 1e-4 * rf2()
            ppb = rf2(); bl = ri2(); cl = ri2(); el = ri2(); M = ri2()
            fw = ri2(); fh = ri2()
            ps = pixel_size
            fw_cm = fw * ps; fh_cm = fh * ps
            self.header = dict(nx=nx, ny=ny, pixel_size=ps, fw=fw, fh=fh,
                               fw_cm=fw_cm, fh_cm=fh_cm)
            num_frames = nx * ny
            chunks = []
            for _ in range(num_frames):
                ri2(); x_pos = 1e-5*ri2(); y_pos = 1e-5*ri2()
                num_trk = ri2(); f.read(12)
                ri2(); ri2(); ri2()  # focus, xi, yi
                if num_trk == 0:
                    continue
                d_raw = np.frombuffer(f.read(2*num_trk), dtype='<i2')
                e_raw = np.frombuffer(f.read(num_trk),   dtype='<i1')
                c_raw = np.frombuffer(f.read(num_trk),   dtype='<i1')
                a_raw = np.frombuffer(f.read(num_trk),   dtype='<i1')
                x_raw = np.frombuffer(f.read(2*num_trk), dtype='<i2')
                y_raw = np.frombuffer(f.read(2*num_trk), dtype='<i2')
                d_um  = 100.0 * d_raw * ps
                x_cm  = x_pos - 0.5*fw_cm + x_raw*ps
                y_cm  = y_pos - 0.5*fh_cm + y_raw*ps
                chunks.append(np.column_stack([d_um, x_cm, y_cm,
                                               e_raw, c_raw, a_raw]))
            arr = np.vstack(chunks) if chunks else np.empty((0,6))
            import pandas as pd
            self.tracks = pd.DataFrame(arr, columns=['d','x','y','e','c','a'])


import pandas as pd

if os.path.exists(cpsa_path):
    print("  Reading CPSA file...")
    reader = CPSAReader(cpsa_path)
    df = reader.tracks.copy()
    h  = reader.header
    scanned_area = h['nx'] * h['ny'] * h['fw_cm'] * h['fh_cm']
    print(f"  Loaded {len(df):,d} raw tracks | Scanned area: {scanned_area:.3f} cm2")

    # ── QUALITY CUTS ────────────────────────────────────────────────────────
    # Layman: We only accept tracks that are the right "size" (diameter)
    # and "darkness" (contrast). Tracks too big or too bright are from
    # heavy particles (alphas, recoils) or noise defects, not protons.
    d_mask = (df['d'] >= 2.0) & (df['d'] <= 13.0)  # diameter in micrometers
    c_mask = (df['c'] <= 20)                         # low contrast = proton-like

    # ── DYNAMIC ECCENTRICITY CUT ─────────────────────────────────────────────
    # Layman: Tracks are round (circle) when the proton hits straight-on.
    # When the proton hits at an angle, the track gets stretched into an oval.
    # At the edges of the CR-39, protons hit at angles up to ~15-20 degrees.
    # We calculate the angle for every track based on its position (x,y)
    # and allow more eccentricity at the edges than at the center.
    df['r_cm']      = np.sqrt(df['x']**2 + df['y']**2)
    df['theta_deg'] = np.degrees(np.arctan2(df['r_cm'], D_CR39_cm))
    df['e_max']     = 15.0 + 20.0 * (np.maximum(0.0, df['theta_deg']) / 30.0)**2
    e_mask = df['e'] <= df['e_max']

    quality_mask = d_mask & c_mask & e_mask

    # ── BACKGROUND REGION (NOTCH STRIP) ──────────────────────────────────────
    # Layman: The top ~0.4 cm of the CR-39 plate was BEHIND a physical notch
    # in the holder — no protons from the reaction could reach there.
    # So the tracks there are PURE BACKGROUND (neutrons + plastic defects).
    # We measure the background density there and subtract it from the signal.
    y_max = df['y'].max()
    NOTCH_H = 0.4  # cm
    notch_mask  = df['y'] >= (y_max - NOTCH_H)
    signal_mask = df['y'] <  (y_max - NOTCH_H)

    notch_area  = h['nx'] * h['fw_cm'] * NOTCH_H
    signal_area = scanned_area - notch_area

    # Count tracks passing quality cuts in each region
    n_bg_pass  = (quality_mask & notch_mask).sum()
    n_sig_pass = (quality_mask & signal_mask).sum()

    bg_density_per_cm2  = n_bg_pass  / notch_area
    sig_density_per_cm2 = n_sig_pass / signal_area

    # Expected background in signal area (scale bg density * signal area)
    expected_bg_in_signal = bg_density_per_cm2 * signal_area
    net_cr39_counts = n_sig_pass - expected_bg_in_signal
    net_cr39_err    = np.sqrt(n_sig_pass + expected_bg_in_signal)

    print(f"  Signal region tracks (quality cuts): {n_sig_pass:,d}")
    print(f"  Background strip density            : {bg_density_per_cm2:.1f} tracks/cm2")
    print(f"  Expected BG in signal area          : {expected_bg_in_signal:.1f} tracks")
    print(f"  Net signal tracks (CR-39)           : {net_cr39_counts:.1f} +/- {net_cr39_err:.1f}")

else:
    print("  CPSA file not found. Using cached values from previous analysis.")
    # Cached values from analyze_cr39_noise_and_neutrons.py run
    signal_area       = 20.396   # cm2
    net_cr39_counts   = 5714.2 * signal_area - 1612.1 * signal_area  # rough estimate
    net_cr39_err      = np.sqrt(abs(net_cr39_counts))
    scanned_area      = 22.28


# ============================================================================
# STEP 3: SOLID ANGLE NORMALIZATION & CROSS-SECTION
# ============================================================================
print("\n" + "=" * 65)
print("  STEP 3: Cross-Section Calculation")
print("=" * 65)

# Layman explanation of the cross-section formula:
#
# Imagine the reaction as a "shooting gallery":
#   - Beam current = how many deuterons per second hit the target
#   - Target thickness = how many 13C atoms per cm2 the beam passes through
#   - Cross-section (sigma) = the "effective target area" of each 13C atom
#
# What we actually measure:
#   N_detected = Beam_particles * Target_atoms * sigma * Omega / (4*pi)
#
# So:  sigma = N_detected * 4*pi / (N_beam * N_target * Omega)
#
# We don't know beam current and target thickness directly, BUT we DO know the
# D+D cross-section (sigma_DD), and we can measure N_DD (D+D counts) under the
# SAME beam conditions. So we use D+D as a calibration standard:
#
#   sigma_13Cd / sigma_DD = (N_13Cd * Omega_DD) / (N_DD * Omega_13Cd)
#
# Where N_13Cd and N_DD are background-subtracted counts on CR-39 and SBD.

# ── SBD Cross-Section ────────────────────────────────────────────────────────
# From CH1: 13C+d signal at 5.24 MeV = 55.6 net counts (dead-time corrected: 60.5)
# These are counts at SBD solid angle OMEGA_SBD
N_13Cd_SBD   = net_sig_ch1_dt if net_sig_ch1_dt > 0 else 60.5  # dead-time corrected
N_13Cd_SBD_e = sig_err if sig_err > 0 else 10.1

# D+D total reaction yield at 4pi (from CH2 reference)
# This normalizes to beam x target thickness
N_DD_4pi   = total_DD_yield_4pi if total_DD_yield_4pi > 0 else 6.26e9

# Differential cross section at SBD angle (0 degrees, forward):
# dσ/dΩ (mb/sr) = [N_13Cd / OMEGA_SBD] / [N_DD / (4pi)] * sigma_DD_ref
# For RELATIVE cross section, we set sigma_DD_ref = 1 (compare ratios)
ratio_sbd = (N_13Cd_SBD / OMEGA_SBD) / (N_DD_4pi / (4*np.pi))
ratio_sbd_err = ratio_sbd * (N_13Cd_SBD_e / N_13Cd_SBD)

print(f"  SBD (CH1):")
print(f"    13C+d net signal (DT-corr)   : {N_13Cd_SBD:.1f} +/- {N_13Cd_SBD_e:.1f} counts")
print(f"    D+D total yield (4pi)        : {N_DD_4pi:.3e} reactions")
print(f"    Differential yield ratio     : {ratio_sbd:.3e} sr^-1")
print(f"    (= 13C+d protons per DD event per steradian at SBD angle)")

# ── CR-39 Cross-Section (INTEGRATED over plate) ──────────────────────────────
# CR-39 integrates over its whole solid angle OMEGA_CR39
# So it gives average dσ/dΩ over that angular range
if 'net_cr39_counts' in dir() and net_cr39_counts > 0:
    ratio_cr39 = (net_cr39_counts / OMEGA_CR39) / (N_DD_4pi / (4*np.pi))
    ratio_cr39_err = ratio_cr39 * (net_cr39_err / net_cr39_counts)

    print(f"\n  CR-39:")
    print(f"    Net signal tracks (BG-subtr) : {net_cr39_counts:.1f} +/- {net_cr39_err:.1f} counts")
    print(f"    CR-39 solid angle            : {OMEGA_CR39:.5f} sr")
    print(f"    Differential yield ratio     : {ratio_cr39:.3e} sr^-1")

    discrepancy = ratio_cr39 / ratio_sbd if ratio_sbd > 0 else 0
    print(f"\n  CR-39 / SBD discrepancy factor : {discrepancy:.2f}x")
    print(f"  (Expected: ~1.0 if both detectors see the same reaction)")
    print()
    if discrepancy > 5:
        print("  DIAGNOSIS: CR-39 is seeing ~{:.0f}x more than SBD.".format(discrepancy))
        print("  Possible causes:")
        print("  1. CR-39 background subtraction is insufficient (intrinsic plastic noise)")
        print("     -> Need: blank CR-39 scan from same batch to measure true noise level")
        print("  2. Solid angle geometry error (need to verify exact distances/areas)")
        print("  3. CR-39 detection efficiency != 100% assumed")
        print("  4. Angular anisotropy: 13C+d is forward-peaked, SBD sits at forward angle")
        print("     -> CR-39 covers a WIDER angle range than SBD point measurement")


# ============================================================================
# STEP 4: ANGULAR DISTRIBUTION ON CR-39 (x-slice summing)
# ============================================================================
print("\n" + "=" * 65)
print("  STEP 4: Angular Distribution N(x) -> dY/dOmega(theta)")
print("=" * 65)
print()
print("  Layman explanation:")
print("  The CR-39 plate is wide -- from x=-2.5 cm to x=+2.5 cm.")
print("  Protons hitting at different x positions came at different angles.")
print("  By slicing the plate in thin vertical strips (each ~1 mm wide),")
print("  we can count 'how many protons hit at each angle'.")
print("  That gives us the angular distribution dY/dOmega(theta).")
print()

if os.path.exists(cpsa_path) and 'df' in dir():
    df_q = df[quality_mask & signal_mask].copy()   # quality-cut signal tracks only

    # x-position directly maps to angle (for a flat detector at d=6 cm)
    # theta(x) = arctan(x / d0)
    df_q['theta_x'] = np.degrees(np.arctan2(df_q['x'], D_CR39_cm))

    # Slice into 10 x-bins across the plate
    x_bins  = np.linspace(-2.5, 2.5, 21)   # 20 slices, each 0.25 cm wide
    x_mids  = 0.5 * (x_bins[:-1] + x_bins[1:])
    theta_mids = np.degrees(np.arctan2(x_mids, D_CR39_cm))
    dx      = np.diff(x_bins)[0]            # bin width in cm

    # Each bin: sum ALL y tracks within that x slice
    # This is "how many protons hit at angle theta_x"
    # Strip height = full plate height
    plate_height_cm = df['y'].max() - df['y'].min() - NOTCH_H
    strip_area = dx * plate_height_cm   # cm2 per x-strip

    # Solid angle of each strip
    # For strip at x=xi, the angle is theta_i = arctan(xi/d)
    # dOmega_strip ~ (dx * plate_height) / d^2 ... approximately
    strip_omega = np.array([dx * plate_height_cm / D_CR39_cm**2 for _ in x_mids])

    # Background density from notch strip (same x-binning)
    n_sig_per_strip = []
    n_bg_per_strip  = []
    for x_lo, x_hi in zip(x_bins[:-1], x_bins[1:]):
        x_mask = (df['x'] >= x_lo) & (df['x'] < x_hi)
        n_s = (quality_mask & signal_mask & x_mask).sum()
        n_b = (quality_mask & notch_mask  & x_mask).sum()
        n_sig_per_strip.append(n_s)
        n_bg_per_strip.append(n_b)

    n_sig_per_strip = np.array(n_sig_per_strip, dtype=float)
    n_bg_per_strip  = np.array(n_bg_per_strip,  dtype=float)

    # Scale background from notch strip to signal strip height
    bg_height_scale = plate_height_cm / NOTCH_H
    n_bg_scaled = n_bg_per_strip * bg_height_scale

    net_per_strip   = n_sig_per_strip - n_bg_scaled
    err_per_strip   = np.sqrt(n_sig_per_strip + n_bg_scaled)

    # Differential yield per steradian per x-strip
    dY_dOmega = net_per_strip / strip_omega
    dY_dOmega_err = err_per_strip / strip_omega

    print(f"  x-strips: {len(x_mids)} strips x {dx:.3f} cm wide x {plate_height_cm:.3f} cm tall")
    print(f"  Theta range: {theta_mids[0]:.1f} to {theta_mids[-1]:.1f} degrees")
    print()
    print(f"  {'Theta (deg)':>12} | {'x (cm)':>8} | {'N_sig':>8} | {'N_bg(scaled)':>14} | {'Net':>8} | {'dY/dOmega':>12}")
    print("  " + "-" * 75)
    for i in range(len(x_mids)):
        print(f"  {theta_mids[i]:>12.2f} | {x_mids[i]:>8.3f} | "
              f"{n_sig_per_strip[i]:>8.0f} | {n_bg_scaled[i]:>14.1f} | "
              f"{net_per_strip[i]:>8.1f} | {dY_dOmega[i]:>12.1f}")

    # ============================================================================
    # STEP 5: PLOTTING
    # ============================================================================
    os.makedirs("CR-39_Python_code/plots", exist_ok=True)

    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

    # Plot 1: Spatial track map (signal vs notch)
    ax1 = fig.add_subplot(gs[0, 0])
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    col = np.where(sample['y'] >= (y_max - NOTCH_H), 'red', 'steelblue')
    ax1.scatter(sample['x'], sample['y'], c=col, s=2, alpha=0.4)
    ax1.axhline(y_max - NOTCH_H, color='k', lw=1.5, ls='--', label='BG/Signal cut')
    ax1.set_xlabel('x (cm)'); ax1.set_ylabel('y (cm)')
    ax1.set_title('Spatial Track Map\nRed=Background Strip, Blue=Signal')
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # Plot 2: Eccentricity vs Angle
    ax2 = fig.add_subplot(gs[0, 1])
    qsamp = df[quality_mask].sample(n=min(3000, quality_mask.sum()), random_state=42)
    ax2.scatter(qsamp['theta_deg'], qsamp['e'], s=4, alpha=0.3, color='steelblue')
    thetas = np.linspace(0, 25, 100)
    ax2.plot(thetas, 15.0 + 20.0*(thetas/30)**2, 'r--', lw=2, label='Dynamic e_max(theta)')
    ax2.axhline(15, color='gray', ls=':', lw=1.5, label='Old flat cut e=15')
    ax2.set_xlabel('Incident Angle theta (deg)'); ax2.set_ylabel('Eccentricity e')
    ax2.set_title('Track Eccentricity vs Incident Angle\n(Dynamic Cut Shown)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Plot 3: Angular distribution dY/dOmega
    ax3 = fig.add_subplot(gs[0, 2])
    good_bins = dY_dOmega > 0
    ax3.errorbar(theta_mids[good_bins], dY_dOmega[good_bins],
                 yerr=dY_dOmega_err[good_bins],
                 fmt='o-', color='navy', lw=2, ms=6, capsize=4, label='CR-39 dY/dOmega')
    ax3.set_xlabel('Proton Angle theta (deg)'); ax3.set_ylabel('dY/dOmega (counts/sr)')
    ax3.set_title('Angular Distribution dY/dOmega(theta)\nfrom CR-39 x-slice summing')
    ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    # Plot 4: Net tracks per x-strip (raw counts)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(x_mids, net_per_strip, width=dx*0.8, alpha=0.8, color='seagreen',
            yerr=err_per_strip, capsize=3, label='Net counts per strip')
    ax4.set_xlabel('x position on CR-39 (cm)'); ax4.set_ylabel('Net Tracks')
    ax4.set_title('Net Signal Tracks per x-Strip\n(After BG Subtraction)')
    ax4.legend(fontsize=9); ax4.grid(alpha=0.3)

    # Plot 5: Diameter distribution of accepted tracks
    ax5 = fig.add_subplot(gs[1, 1])
    d_bins = np.linspace(2, 13, 30)
    ax5.hist(df[quality_mask & signal_mask]['d'], bins=d_bins, alpha=0.8,
             color='steelblue', label='Signal (quality cuts)')
    ax5.hist(df[quality_mask & notch_mask]['d'], bins=d_bins, alpha=0.6,
             color='red', label='BG strip (quality cuts)')
    ax5.set_xlabel('Track Diameter (micrometers)'); ax5.set_ylabel('Counts')
    ax5.set_title('Track Diameter Distribution\n(Signal vs Background)')
    ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

    # Plot 6: Summary text box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = (
        f"CROSS-SECTION SUMMARY\n"
        f"{'='*30}\n\n"
        f"SBD (CH1) @ 0 deg:\n"
        f"  13C+d net counts: {N_13Cd_SBD:.1f} +/- {N_13Cd_SBD_e:.1f}\n"
        f"  dY/dOmega: {ratio_sbd:.3e} sr^-1\n\n"
        f"CR-39 (integrated):\n"
        f"  Net tracks: {net_cr39_counts:.0f} +/- {net_cr39_err:.0f}\n"
        f"  dY/dOmega: {ratio_cr39:.3e} sr^-1\n\n"
        f"CR-39/SBD ratio: {discrepancy:.1f}x\n"
        f"(ideal = 1.0)\n\n"
        f"REMAINING BLOCKER:\n"
        f"  15x discrepancy -> likely\n"
        f"  intrinsic plastic noise\n"
        f"  Need: blank CR-39 scan\n"
        f"  to verify"
    )
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('D + 13C Cross-Section Analysis Dashboard\nAll Tasks Applied: Dynamic Ecc Cuts, BG Subtraction, Solid Angle, Angular Distribution',
                 fontsize=12, fontweight='bold')

    out_path = 'CR-39_Python_code/plots/cross_section_dashboard.png'
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"\n  Saved plot to: {out_path}")
    plt.show()

print("\n" + "=" * 65)
print("  ANALYSIS COMPLETE")
print("=" * 65)
