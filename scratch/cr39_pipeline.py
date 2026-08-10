#!/usr/bin/env python3
# CR-39 Analysis — Integrated Pipeline Script
# Generated automatically from CR39_Analysis_Complete.ipynb & ScanData reader


# --- Cell 02 ---
import os
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

print('Imports OK')



# --- Cell 04 ---
CPSA_FILE = "../CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"
if not os.path.exists(CPSA_FILE) and os.path.exists("CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"):
    CPSA_FILE = "CR-39_data/A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa"

# ── Quality cuts ────────────────────────────────────────────────────────────
D_MIN,  D_MAX  =  2.0, 13.0   # Track diameter bounds (um)
E_MIN,  E_MAX  =  0,   15     # Eccentricity bounds (raw units, int8)
C_MIN,  C_MAX  =  0,   20     # Normal contrast bounds (raw units, int8)
X_MIN,  X_MAX  = -2.5,  2.5   # Spatial x bounds (cm)
Y_MIN,  Y_MAX  = -2.5,  2.5   # Spatial y bounds (cm)

# ── Signal region (rectangular) ────────────────────────────────────────────
SIG_XMIN, SIG_XMAX = -1.5,  1.5
SIG_YMIN, SIG_YMAX = -1.8,  1.4   # A_sig = 9.6 cm^2

# ── Background strip ────────────────────────────────────────────────────────
BG_XMIN,  BG_XMAX  = -1.37,  1.47
BG_YMIN,  BG_YMAX  =  1.50,  1.89

# ── Geometry ────────────────────────────────────────────────────────────────
R_SBD, A_SBD  = 4.0, 0.172    # cm, cm^2 -- SBD at 90 deg to beam
R_CR39        = 6.0            # cm -- CR-39 centered at ~125 deg to beam
THETA0_DEG    = 125.0          # CR-39 piece centre angle w.r.t. beam

# ── Beam energy (corrected: accelerator delivers D2+, so each deuteron
#    carries half the terminal voltage; effective energy after target
#    slowdown is 48 keV -- see HTPD paper) ───────────────────────────────────
E_DEUTERON_NOMINAL_KEV = 62.5
E_EFF_KEV              = 48.0

# ── D+D reference cross-section -- STILL A PLACEHOLDER ────────────────────
# ACTION REQUIRED: replace with an evaluated Bosch-Hale or ENDF/B value for
# D(d,p)T at E_eff = 48 keV. The value below is NOT sourced.
cross_DD          = 0.045   # barn, total (angle-integrated) D(d,p)T
cross_DD_rel_err  = 0.40    # fractional uncertainty, placeholder
differential_cross_DD = cross_DD / (4*np.pi)   # isotropic approximation

# ── R_target -- STILL A PLACEHOLDER, NOT MEASURED ─────────────────────────
# (n*t)_D / (n*t)_13C, implanted-deuterium to carbon-13 areal density ratio.
# See theory doc Section 4 for the saturation-plateau method to measure this.
R_TARGET = 1.0

# ── SBD counts (from the SBD notebook -- 20260413_shot_analysis.ipynb) ────
N_13C_SBD     = 55.6    # counts, sideband-subtracted 5.2 MeV window
N_13C_SBD_err = 9.3     # counts
N_DD_SBD      = 5438178 # counts, DD window (0.7-3 MeV)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

print("Config OK")
print(f"E_deuteron nominal / effective : {E_DEUTERON_NOMINAL_KEV} / {E_EFF_KEV} keV")
print(f"cross_DD (PLACEHOLDER)         : {cross_DD} barn")
print(f"R_TARGET (PLACEHOLDER)         : {R_TARGET}  <-- not measured")


# --- Cell 06 ---
class ScanData:
    """
    Fast reader for CR-39 CPSA binary scan files.

    Parameters
    ----------
    path       : str   – Path to the .cpsa file.
    d_bounds   : (min, max) – Diameter filter in µm.
    e_bounds   : (min, max) – Eccentricity filter (raw int8).
    c_bounds   : (min, max) – Normal contrast filter (raw int8).
    a_bounds   : (min, max) – Average contrast filter (raw int8).
    x_bounds   : (min, max) – Spatial x filter in cm.
    y_bounds   : (min, max) – Spatial y filter in cm.

    Attributes
    ----------
    header  : dict       – Scan metadata (pixel size, frame dimensions, …)
    frames  : DataFrame  – Per-frame info (position, num_tracks, focus, …)
    tracks  : DataFrame  – Track data (d, x, y, e, c, a, frame_number)
    trailer : str        – ASCII trailer appended by the scanner software.
    """

    def __init__(self, path,
                 d_bounds=(0, np.inf),
                 e_bounds=(0, np.inf),
                 c_bounds=(0, np.inf),
                 a_bounds=(0, np.inf),
                 x_bounds=(-np.inf, np.inf),
                 y_bounds=(-np.inf, np.inf)):
        self.header  = {}
        self.frames  = None
        self.tracks  = None
        self.trailer = ''

        with open(path, 'rb') as f:
            self._parse_header(f)
            self._parse_data(f, d_bounds, e_bounds, c_bounds,
                             a_bounds, x_bounds, y_bounds)
            self._parse_trailer(f)

    # ── Helper: read a little-endian int32 ────────────────────────────────────
    @staticmethod
    def _ri(f):
        return struct.unpack('<i', f.read(4))[0]

    @staticmethod
    def _rf(f):
        return struct.unpack('<f', f.read(4))[0]

    # ── Header (48 bytes) ─────────────────────────────────────────────────────
    def _parse_header(self, f):
        ri, rf = self._ri, self._rf
        ps = 1e-4 * rf(f)  # pixel size in cm — read after version/dim fields below

        # Re-read in correct order (version_number is field 0)
        f.seek(0)  # rewind
        h = {
            'version_number':     ri(f),
            'num_x_frames':       ri(f),
            'num_y_frames':       ri(f),
            'num_bins':           ri(f),
            'pixel_size':   1e-4 * rf(f),   # cm per pixel
            'pixels_per_bin':     rf(f),
            'border_limit':       ri(f),
            'contrast_limit':     ri(f),
            'eccentricity_limit': ri(f),
            'M':                  ri(f),
            'frame_width':        ri(f),    # in pixels — converted below
            'frame_height':       ri(f),    # in pixels — converted below
        }
        ps = h['pixel_size']
        h['frame_width']  *= ps   # → cm
        h['frame_height'] *= ps   # → cm
        self.header = h

    # ── Frame + track data ────────────────────────────────────────────────────
    def _parse_data(self, f, d_bounds, e_bounds, c_bounds,
                   a_bounds, x_bounds, y_bounds):
        ps = self.header['pixel_size']
        fw = self.header['frame_width']
        fh = self.header['frame_height']
        num_frames = self.header['num_x_frames'] * self.header['num_y_frames']

        frame_rows = []   # collect frame metadata
        track_chunks = [] # collect filtered track arrays (one ndarray per frame)

        for _ in tqdm(range(num_frames), desc='Reading frames', unit='fr'):
            # ── Frame header (28 bytes) ────────────────────────────────────
            number      = struct.unpack('<i', f.read(4))[0]
            x_pos       = 1e-5 * struct.unpack('<i', f.read(4))[0]  # cm
            y_pos       = 1e-5 * struct.unpack('<i', f.read(4))[0]  # cm
            num_tracks  = struct.unpack('<i', f.read(4))[0]
            f.read(12)   # skip 3 unused int32 fields
            focus       = 1e-2 * struct.unpack('<i', f.read(4))[0]  # µm
            xi          = struct.unpack('<i', f.read(4))[0]          # x index
            yi          = struct.unpack('<i', f.read(4))[0]          # y index

            frame_rows.append((number, x_pos, y_pos, num_tracks,
                               focus, xi, yi))

            if num_tracks == 0:
                continue

            # ── Bulk-read all six track arrays at once ─────────────────────
            # Layout in file: d[n], e[n], c[n], a[n], x[n], y[n]
            d_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 → µm after scale
            e_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  eccentricity
            c_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  normal contrast
            a_raw = np.frombuffer(f.read(num_tracks),     dtype='<i1')  # int8  average contrast
            x_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 pixel position
            y_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')  # int16 pixel position

            # ── Unit conversion (vectorised) ───────────────────────────────
            d_um = 100.0 * d_raw * ps         # diameter in µm
            x_cm = x_pos - 0.5*fw + x_raw*ps  # absolute x in cm
            y_cm = y_pos - 0.5*fh + y_raw*ps  # absolute y in cm

            # ── Vectorised quality + spatial filter ────────────────────────
            mask = (
                (d_um  >= d_bounds[0]) & (d_um  <= d_bounds[1]) &
                (e_raw >= e_bounds[0]) & (e_raw <= e_bounds[1]) &
                (c_raw >= c_bounds[0]) & (c_raw <= c_bounds[1]) &
                (a_raw >= a_bounds[0]) & (a_raw <= a_bounds[1]) &
                (x_cm  >= x_bounds[0]) & (x_cm  <= x_bounds[1]) &
                (y_cm  >= y_bounds[0]) & (y_cm  <= y_bounds[1])
            )

            n_pass = mask.sum()
            if n_pass == 0:
                continue

            # Pack selected tracks into a (n_pass × 7) float64 array
            chunk = np.empty((n_pass, 7), dtype=np.float64)
            chunk[:, 0] = number          # frame_number (stored as float, cast later)
            chunk[:, 1] = d_um[mask]      # d  [µm]
            chunk[:, 2] = x_cm[mask]      # x  [cm]
            chunk[:, 3] = y_cm[mask]      # y  [cm]
            chunk[:, 4] = e_raw[mask]     # e  (eccentricity)
            chunk[:, 5] = c_raw[mask]     # c  (normal contrast)
            chunk[:, 6] = a_raw[mask]     # a  (average contrast)
            track_chunks.append(chunk)

        # ── Build DataFrames once at the end (avoids costly pd.concat in loop) ─
        self.frames = pd.DataFrame(frame_rows, columns=[
            'number', 'x_position', 'y_position', 'num_tracks',
            'focus', 'x_position_index', 'y_position_index'
        ])

        if track_chunks:
            arr = np.vstack(track_chunks)
            self.tracks = pd.DataFrame(arr, columns=[
                'frame_number', 'd', 'x', 'y', 'e', 'c', 'a'
            ])
            # Restore compact dtypes to save memory
            self.tracks['frame_number'] = self.tracks['frame_number'].astype(np.int32)
            self.tracks[['e', 'c', 'a']] = self.tracks[['e', 'c', 'a']].astype(np.int8)
            self.tracks[['d', 'x', 'y']] = self.tracks[['d', 'x', 'y']].astype(np.float32)
        else:
            self.tracks = pd.DataFrame(
                columns=['frame_number', 'd', 'x', 'y', 'e', 'c', 'a'])

    # ── Trailer (ASCII metadata appended by scanner) ──────────────────────────
    def _parse_trailer(self, f):
        f.read(4)  # skip 4-byte separator
        self.trailer = f.read().decode('latin-1')

    def __repr__(self):
        h = self.header
        return (
            f'ScanData  {h["num_x_frames"]}×{h["num_y_frames"]} frames  '
            f'pixel={h["pixel_size"]*1e4:.4f} µm  '
            f'tracks={len(self.tracks):,}'
        )


# --- Cell 08 ---
# ── Quality-cut load (used for the final, cut track set) ──────────────────
data = ScanData(
    CPSA_FILE,
    d_bounds=(D_MIN, D_MAX),
    e_bounds=(E_MIN, E_MAX),
    c_bounds=(C_MIN, C_MAX),
    x_bounds=(X_MIN, X_MAX),
    y_bounds=(Y_MIN, Y_MAX),
)
tracks = data.tracks
frames = data.frames
h = data.header

print(data)
print(f"Scan grid  : {h['num_x_frames']} x {h['num_y_frames']} = "
      f"{h['num_x_frames']*h['num_y_frames']:,} frames")
print(f"Tracks loaded (after cuts): {len(tracks):,}")

# ── Raw load, no quality cuts (needed for the background-subtraction cell,
#    which must see the un-cut population before comparing signal vs junk) ──
data_raw = ScanData(CPSA_FILE, x_bounds=(X_MIN, X_MAX), y_bounds=(Y_MIN, Y_MAX))
tr = data_raw.tracks
print(f"Raw tracks (no quality cuts): {len(tr):,}")


# --- Cell 10 ---
fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
H, xedges, yedges = np.histogram2d(tr['d'], tr['c'], bins=[150, 100],
                                    range=[[0, 25], [0, 100]])
xc = 0.5 * (xedges[:-1] + xedges[1:])
yc = 0.5 * (yedges[:-1] + yedges[1:])
if H.max() > 50:
    levels = np.logspace(np.log10(50), np.log10(H.max()), 10)
    ax.contour(xc, yc, H.T, levels=levels, cmap='plasma', linewidths=1.5)
ax.axvline(D_MIN, color='lime', ls='--', lw=1.2, label=f'd_min={D_MIN}')
ax.axvline(D_MAX, color='cyan', ls='--', lw=1.2, label=f'd_max={D_MAX}')
ax.axhline(C_MAX, color='red',  ls='--', lw=1.2, label=f'c_max={C_MAX}')
ax.set_xlabel('Track Diameter (um)')
ax.set_ylabel('Track Contrast (%)')
ax.legend(fontsize=9)
ax.set_xlim(0, 25); ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/01_precut_diameter_vs_contrast.png', dpi=200)
# plt.show()


# --- Cell 12 ---
tr_e = tr[(tr['e'] >= E_MIN) & (tr['e'] <= E_MAX)]
print(f"Raw tracks: {len(tr):,} -> after eccentricity cut only (e <= {E_MAX}): {len(tr_e):,}")

sig_mask_e = (
    (tr_e['x'] >= SIG_XMIN) & (tr_e['x'] <= SIG_XMAX) &
    (tr_e['y'] >= SIG_YMIN) & (tr_e['y'] <= SIG_YMAX)
)
bg_mask_e = (
    (tr_e['x'] >= BG_XMIN) & (tr_e['x'] <= BG_XMAX) &
    (tr_e['y'] >= BG_YMIN) & (tr_e['y'] <= BG_YMAX)
)
sig_tracks_e = tr_e[sig_mask_e]
bg_tracks_e  = tr_e[bg_mask_e]

A_sig = (SIG_XMAX - SIG_XMIN) * (SIG_YMAX - SIG_YMIN)
A_bg  = (BG_XMAX  - BG_XMIN)  * (BG_YMAX  - BG_YMIN)
scale = A_sig / A_bg

cd_bins = [np.linspace(0, 25, 100), np.linspace(0, 100, 100)]
H_sig, d_edges, c_edges = np.histogram2d(sig_tracks_e['d'], sig_tracks_e['c'], bins=cd_bins)
H_bg,  _,       _       = np.histogram2d(bg_tracks_e['d'],  bg_tracks_e['c'],  bins=cd_bins)
H_bg_scaled  = H_bg * scale
H_subtracted = H_sig - H_bg_scaled

d_centers = 0.5 * (d_edges[:-1] + d_edges[1:])
c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
DD, CC = np.meshgrid(d_centers, c_centers, indexing='ij')
cut_mask = (DD >= D_MIN) & (DD <= D_MAX) & (CC <= C_MAX)

N_NET_PRIMARY = float(H_subtracted[cut_mask].sum())
N_sig_stat = float(H_sig[cut_mask].sum())
N_bg_stat  = float(H_bg[cut_mask].sum())
sigma_stat = float(np.sqrt(N_sig_stat + (scale**2) * N_bg_stat))

print("="*70)
print("  NET SIGNAL (C-vs-D heatmap subtraction, meeting-3 ordering)")
print("="*70)
print(f"  Area scale factor A_sig/A_bg : {scale:.4f}")
print(f"  N_net                        : {N_NET_PRIMARY:,.1f}")
print(f"  Statistical uncertainty      : +/- {sigma_stat:,.1f}")
print("="*70)

# Three-panel plot: signal / scaled background / subtracted
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
for ax, Hmap, title in zip(
    axes, [H_sig, H_bg_scaled, H_subtracted],
    [f'Signal region (N={H_sig.sum():.0f})',
     f'Background x{scale:.2f} (N={H_bg_scaled.sum():.0f})',
     f'Subtracted (N={H_subtracted.sum():.0f})']
):
    im = ax.pcolormesh(d_edges, c_edges, Hmap.T, cmap='RdBu_r',
                        vmin=-np.abs(Hmap).max()*0.3, vmax=np.abs(Hmap).max()*0.3)
    ax.axvline(D_MIN, color='cyan', ls='--', lw=1)
    ax.axvline(D_MAX, color='cyan', ls='--', lw=1)
    ax.axhline(C_MAX, color='lime', ls='--', lw=1)
    ax.set_xlabel('Diameter (um)'); ax.set_title(title, fontsize=10)
axes[0].set_ylabel('Contrast (%)')
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/02_signal_background_subtracted.png', dpi=200)
# plt.show()


# --- Cell 14 ---
in_x = (tr_e['x'] >= SIG_XMIN) & (tr_e['x'] <= SIG_XMAX)
y_vals = tr_e['y'][in_x]

y_edges_strip = np.arange(-2.5, 2.5 + 0.1, 0.1)
counts_per_slice, _ = np.histogram(y_vals, bins=y_edges_strip)
y_mid = 0.5 * (y_edges_strip[:-1] + y_edges_strip[1:])

R_PIECE = 2.5   # cm, CR-39 physical radius
x_width = SIG_XMAX - SIG_XMIN
half_chord = np.sqrt(np.clip(R_PIECE**2 - y_mid**2, 0, None))
slice_width = np.minimum(x_width, 2*half_chord)
slice_area = slice_width * 0.1   # cm^2, 0.1 cm slice height
slice_area[slice_area <= 0] = np.nan
density = counts_per_slice / slice_area

sig_slice = (y_mid >= SIG_YMIN) & (y_mid <= SIG_YMAX)
notch_slice = (y_mid >= BG_YMIN) & (y_mid <= BG_YMAX)
above_notch_slice = y_mid > BG_YMAX

density_box   = np.nanmean(density[sig_slice])
density_notch = np.nanmean(density[notch_slice])
density_above = np.nanmean(density[above_notch_slice]) if above_notch_slice.any() else np.nan

print("="*70)
print("  IS THE NOTCH CLEAN?")
print("="*70)
print(f"  Average density inside signal box : {density_box:>10,.0f} tracks/cm2")
print(f"  Average density inside notch      : {density_notch:>10,.0f} tracks/cm2")
print(f"  Average density above the notch   : {density_above:>10,.0f} tracks/cm2")
print("-"*70)
print(f"  Notch / box ratio                 : {density_notch/density_box:>10.3f}")
print("  (~0 would mean a clean notch; ~1 means the notch is as busy as signal)")
print("="*70)

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.plot(y_mid, density, '-', color='steelblue', lw=1.5)
ax.axvspan(SIG_YMIN, SIG_YMAX, color='red', alpha=0.15, label='Signal box')
ax.axvspan(BG_YMIN, BG_YMAX, color='blue', alpha=0.15, label='Background notch')
ax.set_xlabel('y (cm)'); ax.set_ylabel('Track density (tracks/cm^2)')
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f'{PLOT_DIR}/03_notch_contamination_check.png', dpi=200)
# plt.show()


# --- Cell 16 ---
d0_cm = R_CR39
t_df = tr.copy()
t_df['r_cm']      = np.sqrt(t_df['x']**2 + t_df['y']**2)
t_df['theta_deg'] = np.degrees(np.arctan2(t_df['r_cm'], d0_cm))

def eccentricity_limit_conservative(theta_deg):
    angle_pts = np.array([0, 10, 20, 25, 30])
    e_pts     = np.array([3,  7, 25, 33, 34])   # upper envelope, Przybocki 2.9 MeV curve
    return np.interp(theta_deg, angle_pts, e_pts)

t_df['e_limit_dynamic']  = eccentricity_limit_conservative(t_df['theta_deg'])
t_df['pass_dynamic_ecc'] = t_df['e'] <= t_df['e_limit_dynamic']

n_flat = (t_df['e'] <= E_MAX).sum()
n_dyn  = t_df['pass_dynamic_ecc'].sum()

print(f"Incidence angle range across the piece : {t_df['theta_deg'].min():.2f} - {t_df['theta_deg'].max():.2f} deg")
print(f"Flat cut (e <= {E_MAX})            : {n_flat:,} tracks pass")
print(f"Angle-dependent cut (conservative) : {n_dyn:,} tracks pass  ({n_dyn - n_flat:+,})")
print()
print("NOTE: not yet wired into the Section 6 net-signal number above -- this")
print("cell characterizes the effect only. Wiring it in requires the full")
print("per-track energy calibration (Przybocki two-parameter model) first.")


# --- Cell 18 ---
# ── PART A ──────────────────────────────────────────────────────────────────
Y_ratio = N_13C_SBD / N_DD_SBD
Y_ratio_relerr = N_13C_SBD_err / N_13C_SBD   # DD count effectively exact (huge N)

differential_cross_13CD = Y_ratio * (1.0 / R_TARGET) * differential_cross_DD
differential_cross_13CD_relerr = float(np.sqrt(Y_ratio_relerr**2 + cross_DD_rel_err**2))

print("="*70)
print("  PART A: 13C(d,p)14C DIFFERENTIAL CROSS-SECTION (SBD ratio method)")
print("="*70)
print(f"  Y_13C/Y_DD (SBD)         : {Y_ratio:.4e}  +/- {Y_ratio_relerr*100:.1f}%")
print(f"  R_target                : {R_TARGET}  <-- PLACEHOLDER, not measured")
print(f"  dsigma/dOmega (13C)      : {differential_cross_13CD:.4e} barn/sr")
print(f"                             +/- {differential_cross_13CD_relerr*100:.1f}%")
print("  ** NOT A FINAL, REPORTABLE NUMBER ** -- cross_DD and R_target are")
print("  both explicit placeholders (Section 2). Replace before publishing.")
print("="*70)
print()

# ── PART B ──────────────────────────────────────────────────────────────────
def expected_cr39_flux_from_sbd(C_sbd, A_sbd, r_sbd, r_cr39,
                                 theta_deg=0.0, facility_factor=1.0,
                                 background_sbd=None):
    """
    Phi_expected = (C_sbd/A_sbd) * (r_sbd/r_cr39)^2 * cos(theta) * facility_factor
    facility_factor disabled (=1.0) per meeting-3: the 0.9676 value was
    measured for 115 deg, not our ~125 deg CR-39 -- do not re-enable without
    a properly re-derived angular correction (theory doc Section 7/9).
    """
    C = C_sbd - background_sbd if background_sbd is not None else C_sbd
    return (C / A_sbd) * (r_sbd / r_cr39)**2 * np.cos(np.radians(theta_deg)) * facility_factor

expected_flux_at_cr39 = expected_cr39_flux_from_sbd(
    C_sbd=N_13C_SBD, A_sbd=A_SBD, r_sbd=R_SBD, r_cr39=R_CR39,
    theta_deg=0.0, facility_factor=1.0,
)
expected_count_at_cr39 = expected_flux_at_cr39 * A_sig
agreement_ratio = N_NET_PRIMARY / expected_count_at_cr39

print("="*70)
print("  PART B: CR-39 CROSS-CHECK (independent validation)")
print("="*70)
print(f"  CR-39 observed net signal        : {N_NET_PRIMARY:,.1f}")
print(f"  CR-39 expected (from SBD flux)   : {expected_count_at_cr39:,.1f}")
print(f"  Agreement ratio (observed/expect): {agreement_ratio:.2f}x")
print("="*70)
print()
print("="*70)
print("  KNOWN OPEN ITEMS (see Section 2 and 7 above)")
print("="*70)
print(f"  - Notch/box density ratio        : {density_notch/density_box:.3f} (background may not be clean)")
print("  - cross_DD is an unsourced placeholder")
print("  - R_target = 1 is assumed, not measured")
print("  - Angular correction is disabled (facility_factor=1.0), not replaced")
print("="*70)

