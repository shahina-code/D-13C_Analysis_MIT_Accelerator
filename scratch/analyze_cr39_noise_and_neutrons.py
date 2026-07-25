import os
import struct
import numpy as np
import pandas as pd

# Load CPSA file using python ScanData parser
cpsa_path = os.path.join(".", "CR-39_data", "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")
if not os.path.exists(cpsa_path):
    cpsa_path = os.path.join("..", "CR-39_data", "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")

class CPSAReader:
    def __init__(self, path):
        with open(path, 'rb') as f:
            ri = lambda: struct.unpack('<i', f.read(4))[0]
            rf = lambda: struct.unpack('<f', f.read(4))[0]
            
            f.seek(0)
            self.header = {
                'version': ri(),
                'nx': ri(),
                'ny': ri(),
                'nbins': ri(),
                'pixel_size': 1e-4 * rf(), # cm
                'pixels_per_bin': rf(),
                'border_limit': ri(),
                'contrast_limit': ri(),
                'ecc_limit': ri(),
                'M': ri(),
                'fw': ri(), # px
                'fh': ri()  # px
            }
            ps = self.header['pixel_size']
            fw = self.header['fw'] * ps # cm
            fh = self.header['fh'] * ps # cm
            self.header['fw_cm'] = fw
            self.header['fh_cm'] = fh
            
            num_frames = self.header['nx'] * self.header['ny']
            
            track_chunks = []
            for _ in range(num_frames):
                number = struct.unpack('<i', f.read(4))[0]
                x_pos = 1e-5 * struct.unpack('<i', f.read(4))[0]
                y_pos = 1e-5 * struct.unpack('<i', f.read(4))[0]
                num_tracks = struct.unpack('<i', f.read(4))[0]
                f.read(12)
                focus = 1e-2 * struct.unpack('<i', f.read(4))[0]
                xi = struct.unpack('<i', f.read(4))[0]
                yi = struct.unpack('<i', f.read(4))[0]
                
                if num_tracks == 0:
                    continue
                    
                d_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')
                e_raw = np.frombuffer(f.read(num_tracks), dtype='<i1')
                c_raw = np.frombuffer(f.read(num_tracks), dtype='<i1')
                a_raw = np.frombuffer(f.read(num_tracks), dtype='<i1')
                x_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')
                y_raw = np.frombuffer(f.read(2 * num_tracks), dtype='<i2')
                
                d_um = 100.0 * d_raw * ps
                x_cm = x_pos - 0.5*fw + x_raw*ps
                y_cm = y_pos - 0.5*fh + y_raw*ps
                
                chunk = np.column_stack([d_um, x_cm, y_cm, e_raw, c_raw, a_raw])
                track_chunks.append(chunk)
                
            if track_chunks:
                arr = np.vstack(track_chunks)
                self.tracks = pd.DataFrame(arr, columns=['d', 'x', 'y', 'e', 'c', 'a'])
            else:
                self.tracks = pd.DataFrame(columns=['d', 'x', 'y', 'e', 'c', 'a'])

print("Reading CPSA scan file...")
reader = CPSAReader(cpsa_path)
df = reader.tracks

# Calculate Scanned Area
h = reader.header
scanned_area_cm2 = h['nx'] * h['ny'] * h['fw_cm'] * h['fh_cm']
tot_tracks = len(df)
raw_density = tot_tracks / scanned_area_cm2

print(f"CR-39 Scanned Area: {scanned_area_cm2:.4f} cm^2 ({h['nx']}x{h['ny']} frames)")
print(f"Total Raw Tracks  : {tot_tracks:,d}")
print(f"Raw Track Density : {raw_density:,.1f} tracks/cm^2")
print("-" * 60)

# Track Density vs Contrast c (bins of 5%)
print("\n=== TRACK DENSITY vs OPTICAL CONTRAST (c) ===")
print(f"{'Contrast Range (%)':<20} | {'Track Count':<12} | {'Density (tracks/cm^2)':<22}")
print("-" * 60)

c_bins = np.arange(0, 55, 5)
for i in range(len(c_bins)-1):
    c_low, c_high = c_bins[i], c_bins[i+1]
    sub = df[(df['c'] >= c_low) & (df['c'] < c_high)]
    cnt = len(sub)
    dens = cnt / scanned_area_cm2
    print(f"{c_low:2d}% to {c_high:2d}%            | {cnt:>12,d} | {dens:>22,.1f}")

print("=" * 60)

# Incident Angle & Dynamic Cut Effect Analysis
d0_cm = 6.0
df['r_cm'] = np.sqrt(df['x']**2 + df['y']**2)
df['theta_deg'] = np.degrees(np.arctan2(df['r_cm'], d0_cm))
df['e_max'] = 15.0 + 20.0 * (np.maximum(0.0, df['theta_deg']) / 30.0)**2

# Cuts
d_mask = (df['d'] >= 2.0) & (df['d'] <= 13.0)
c_mask = (df['c'] <= 20)

flat_mask = d_mask & c_mask & (df['e'] <= 15.0)
dyn_mask  = d_mask & c_mask & (df['e'] <= df['e_max'])

n_flat = flat_mask.sum()
n_dyn = dyn_mask.sum()
dens_flat = n_flat / scanned_area_cm2
dens_dyn = n_dyn / scanned_area_cm2

print("\n=== EFFECT OF INCIDENT ANGLE ECCENTRICITY CUT ===")
print(f"Flat cut (e <= 15.0)               : {n_flat:,d} tracks ({dens_flat:,.1f} tracks/cm^2)")
print(f"Dynamic cut e <= e_max(theta)      : {n_dyn:,d} tracks ({dens_dyn:,.1f} tracks/cm^2)")
print(f"Recovered Edge Tracks              : +{n_dyn - n_flat:,d} tracks (+{(n_dyn - n_flat)/n_flat*100:.2f}%)")
print("=" * 60)

# NEUTRON FLUENCE & EFFICIENCY CALCULATION (Frenje et al. 2002)
# SBD D+D total yield N_DD_SBD = 5.35e6
# SBD solid angle = 0.172 / 16.0 = 0.01075 sr
# Total DD neutrons emitted into 4pi: Y_DD = N_DD_SBD / (omega_SBD / 4pi)
Y_DD = 5.354918e6 / (0.01075 / (4 * np.pi))
phi_n_CR39 = Y_DD / (4 * np.pi * (6.0**2))
eff_n_frenje = 1.1e-4 # tracks per neutron
rho_n_direct = phi_n_CR39 * eff_n_frenje
rho_n_backscatter = 90.0 # tracks/cm^2 from Frenje et al 2002

print("\n=== DD NEUTRON VS INTRINSIC PLASTIC NOISE DIAGNOSTIC ===")
print(f"Total Chamber DD Neutrons Generated (Y_DD) : {Y_DD:.3e} neutrons")
print(f"DD Neutron Fluence at CR-39 (r = 6 cm)      : {phi_n_CR39:.3e} neutrons/cm^2")
print(f"Frenje et al. (2002) Front Efficiency (eps) : 1.1 x 10^-4 tracks/neutron")
print(f"Expected Direct DD Neutron Track Density   : {rho_n_direct:.1f} tracks/cm^2")
print(f"Expected Backscattered Neutron Density    : ~{rho_n_backscatter:.1f} tracks/cm^2")
print(f"TOTAL Expected Neutron Background          : ~{rho_n_direct + rho_n_backscatter:.1f} tracks/cm^2")
print("-" * 60)
print(f"Observed CR-39 Track Density (Signal Box)   : {dens_dyn:,.1f} tracks/cm^2")
print(f"Excess Background Density over Neutrons     : {dens_dyn - (rho_n_direct + rho_n_backscatter):,.1f} tracks/cm^2")
print("=" * 60)
