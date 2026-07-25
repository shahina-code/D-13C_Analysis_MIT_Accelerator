"""
CR-39 Background Strip Analysis
================================
Task from Shahina (July 25 meeting):
  "In the background strip (notch region), look at track density vs contrast (c),
   see if numbers are around ~300 tracks/cm² (intrinsic plastic noise level from
   Frenje 2002, Fig.5) or higher (indicating neutron background)."

This script reads the CPSA file, isolates the background notch strip
(top ~0.4 cm of detector plate), plots track density vs contrast, and
compares against expected intrinsic noise level.
"""

import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 1. File path ──────────────────────────────────────────────────────────────
cpsa_path = os.path.join(".", "CR-39_data",
    "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")
if not os.path.exists(cpsa_path):
    cpsa_path = os.path.join("..", "CR-39_data",
        "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")

if not os.path.exists(cpsa_path):
    raise FileNotFoundError(f"CPSA file not found at: {cpsa_path}\n"
                            f"Please run from the MAIN project directory.")


# ── 2. CPSA Reader ─────────────────────────────────────────────────────────────
class CPSAReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            ri = lambda: struct.unpack('<i', f.read(4))[0]
            rf = lambda: struct.unpack('<f', f.read(4))[0]
            self.header = {
                'version': ri(), 'nx': ri(), 'ny': ri(), 'nbins': ri(),
                'pixel_size': 1e-4 * rf(),   # cm
                'pixels_per_bin': rf(),
                'border_limit': ri(), 'contrast_limit': ri(),
                'ecc_limit': ri(), 'M': ri(), 'fw': ri(), 'fh': ri()
            }
            ps  = self.header['pixel_size']
            fw  = self.header['fw'] * ps   # cm per frame
            fh  = self.header['fh'] * ps
            self.header.update({'fw_cm': fw, 'fh_cm': fh})

            num_frames = self.header['nx'] * self.header['ny']
            chunks = []
            for _ in range(num_frames):
                number    = struct.unpack('<i', f.read(4))[0]
                x_pos     = 1e-5 * struct.unpack('<i', f.read(4))[0]
                y_pos     = 1e-5 * struct.unpack('<i', f.read(4))[0]
                num_trk   = struct.unpack('<i', f.read(4))[0]
                f.read(12)
                focus     = 1e-2  * struct.unpack('<i', f.read(4))[0]
                xi        = struct.unpack('<i', f.read(4))[0]
                yi        = struct.unpack('<i', f.read(4))[0]
                if num_trk == 0:
                    continue
                d_raw = np.frombuffer(f.read(2 * num_trk), dtype='<i2')
                e_raw = np.frombuffer(f.read(num_trk),     dtype='<i1')
                c_raw = np.frombuffer(f.read(num_trk),     dtype='<i1')
                a_raw = np.frombuffer(f.read(num_trk),     dtype='<i1')
                x_raw = np.frombuffer(f.read(2 * num_trk), dtype='<i2')
                y_raw = np.frombuffer(f.read(2 * num_trk), dtype='<i2')

                d_um  = 100.0 * d_raw * ps
                x_cm  = x_pos - 0.5 * fw + x_raw * ps
                y_cm  = y_pos - 0.5 * fh + y_raw * ps
                chunks.append(np.column_stack([d_um, x_cm, y_cm, e_raw, c_raw, a_raw]))

            arr = np.vstack(chunks) if chunks else np.empty((0, 6))
            self.tracks = pd.DataFrame(arr, columns=['d','x','y','e','c','a'])


# ── 3. Load ───────────────────────────────────────────────────────────────────
print("Reading CPSA file…")
reader = CPSAReader(cpsa_path)
df = reader.tracks.copy()
h  = reader.header

scanned_area_cm2 = h['nx'] * h['ny'] * h['fw_cm'] * h['fh_cm']
y_max = df['y'].max()
y_min = df['y'].min()
plate_height = y_max - y_min

# ── 4. Define Regions ─────────────────────────────────────────────────────────
# Background notch strip: top 0.4 cm of detector (highest y values)
NOTCH_HEIGHT = 0.4   # cm  ← matches the notch geometry from action_plan.txt
notch_y_min  = y_max - NOTCH_HEIGHT
notch_area   = h['nx'] * h['fw_cm'] * NOTCH_HEIGHT

# Signal box: the rest of the plate
signal_area  = scanned_area_cm2 - notch_area

bg_mask  = df['y'] >= notch_y_min
sig_mask = df['y'] <  notch_y_min

df_bg  = df[bg_mask].copy()
df_sig = df[sig_mask].copy()

# ── 5. Track Density vs Contrast in Background Strip ─────────────────────────
print(f"\nCR-39 Detector Summary")
print(f"  Plate Y range     : {y_min:.3f} – {y_max:.3f} cm (height {plate_height:.3f} cm)")
print(f"  Scanned Area      : {scanned_area_cm2:.3f} cm²")
print(f"  Notch strip       : y >= {notch_y_min:.3f} cm, area = {notch_area:.3f} cm²")
print(f"  Signal region     : y <  {notch_y_min:.3f} cm, area = {signal_area:.3f} cm²")
print(f"  Total tracks      : {len(df):,d}  |  Background strip: {len(df_bg):,d}  |  Signal: {len(df_sig):,d}")
print()

c_bins  = np.arange(0, 80, 5)
bg_densities  = []
sig_densities = []
contrast_mids = []

print(f"{'Contrast (%)':<16} | {'BG Density (t/cm²)':<22} | {'Sig Density (t/cm²)':<22} | {'BG/Sig ratio':<12}")
print("-" * 80)
for i in range(len(c_bins) - 1):
    lo, hi = c_bins[i], c_bins[i+1]
    n_bg  = ((df_bg['c']  >= lo) & (df_bg['c']  < hi)).sum()
    n_sig = ((df_sig['c'] >= lo) & (df_sig['c'] < hi)).sum()
    d_bg  = n_bg  / notch_area  if notch_area  > 0 else 0
    d_sig = n_sig / signal_area if signal_area > 0 else 0
    ratio = d_bg / d_sig if d_sig > 0 else float('nan')
    bg_densities.append(d_bg)
    sig_densities.append(d_sig)
    contrast_mids.append((lo + hi) / 2)
    print(f"  {lo:2d}% – {hi:2d}%       | {d_bg:>18,.1f}  | {d_sig:>18,.1f}  | {ratio:>12.3f}")

print("-" * 80)
total_bg_density  = len(df_bg)  / notch_area
total_sig_density = len(df_sig) / signal_area
print(f"  TOTAL ALL c        | {total_bg_density:>18,.1f}  | {total_sig_density:>18,.1f}  |")
print()

# ── 6. Intrinsic Noise Reference ──────────────────────────────────────────────
INTRINSIC_NOISE_REF = 300.0  # tracks/cm²  (Frenje et al. 2002, Fig. 5 nominal)
print(f"Reference intrinsic plastic noise (Frenje 2002, Fig.5):  ~{INTRINSIC_NOISE_REF:.0f} tracks/cm²")
print(f"Observed background strip total density              :  {total_bg_density:.1f} tracks/cm²")
excess_bg = max(0, total_bg_density - INTRINSIC_NOISE_REF)
print(f"Excess above intrinsic noise level                   :  {excess_bg:.1f} tracks/cm²")
if excess_bg > 0:
    print(f"  >> Possible neutron or environmental background: {excess_bg:.1f} tracks/cm2")
else:
    print(f"  >> Background strip is consistent with intrinsic plastic noise only.")

print()
print("DD Neutron expected contribution (from report):  ~1,612 tracks/cm2")
print(f"Total background strip density:                  {total_bg_density:.1f} tracks/cm2")
if total_bg_density < 1612:
    print("  >> BG strip BELOW expected neutron level -- geometry or efficiency may differ.")
elif total_bg_density >= 1612:
    print("  >> BG strip ABOVE expected neutron level -- neutron background is real contributor.")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), dpi=150)
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

contrast_mids = np.array(contrast_mids)

# Plot 1: Background vs Contrast
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(contrast_mids, bg_densities, width=4, alpha=0.8, color='tomato', label='Background Strip')
ax1.axhline(INTRINSIC_NOISE_REF, color='navy', lw=2, ls='--', label=f'Intrinsic Noise Ref ({INTRINSIC_NOISE_REF:.0f}/cm²)')
ax1.set_xlabel('Optical Contrast c (%)')
ax1.set_ylabel('Track Density (tracks/cm²)')
ax1.set_title('Background Strip: Track Density vs Contrast\n(Notch Region — Expected to be ~300 t/cm² intrinsic noise)')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

# Plot 2: Signal Region vs Contrast
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(contrast_mids, sig_densities, width=4, alpha=0.8, color='steelblue', label='Signal Region')
ax2.set_xlabel('Optical Contrast c (%)')
ax2.set_ylabel('Track Density (tracks/cm²)')
ax2.set_title('Signal Region: Track Density vs Contrast')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Plot 3: BG vs Signal overlay
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(contrast_mids, bg_densities,  'r-o', lw=2, ms=5, label='Background Strip')
ax3.plot(contrast_mids, sig_densities, 'b-s', lw=2, ms=5, label='Signal Region')
ax3.axhline(INTRINSIC_NOISE_REF, color='gray', lw=1.5, ls=':', label='Intrinsic Noise Ref')
ax3.set_xlabel('Optical Contrast c (%)')
ax3.set_ylabel('Track Density (tracks/cm²)')
ax3.set_title('BG Strip vs Signal Region: Track Density vs Contrast')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: 2D Spatial Map colored by Y position (BG vs signal)
ax4 = fig.add_subplot(gs[1, 1])
sample = df.sample(n=min(8000, len(df)), random_state=42)
colors = np.where(sample['y'] >= notch_y_min, 'red', 'steelblue')
ax4.scatter(sample['x'], sample['y'], c=colors, s=2, alpha=0.5)
ax4.axhline(notch_y_min, color='black', lw=1.5, ls='--', label=f'BG/Signal cut at y={notch_y_min:.3f} cm')
ax4.set_xlabel('Detector X (cm)')
ax4.set_ylabel('Detector Y (cm)')
ax4.set_title('Spatial Map: Background Strip (red) vs Signal (blue)')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

plt.suptitle('CR-39 Background Strip vs Signal Analysis\n(Task from Shahina, July 25 Meeting)',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig('CR-39_Python_code/plots/bg_strip_vs_signal_analysis.png',
            bbox_inches='tight', dpi=150)
print("\nPlot saved to: CR-39_Python_code/plots/bg_strip_vs_signal_analysis.png")
plt.show()
