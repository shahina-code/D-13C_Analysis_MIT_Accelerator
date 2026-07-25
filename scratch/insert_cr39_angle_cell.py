import json

nb_path = "CR-39_Python_code/CR39_Analysis_Optimized.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new cell for Incident Angle & Dynamic Eccentricity Cut
new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# ==============================================================================\n",
  "# ?? INCIDENT ANGLE & POSITION-DEPENDENT ECCENTRICITY CUT (Figure 9b Model)\n",
  "# ==============================================================================\n",
  "# Physics Background:\n",
  "# For a flat CR-39 detector at distance d0 = 6.0 cm from the target (origin),\n",
  "# particles hitting at off-normal positions (x, y) enter the plastic at an angle:\n",
  "#   theta_inc(x, y) = arctan( sqrt(x^2 + y^2) / d0 )  [degrees]\n",
  "# As shown in literature (Fig. 9b), non-normal incidence naturally elongates track\n",
  "# shapes, increasing eccentricity 'e' with angle theta:\n",
  "#   - theta = 0 deg  --> e ~ 0 - 3  (normal incidence)\n",
  "#   - theta = 10 deg --> e ~ 5\n",
  "#   - theta = 20 deg --> e ~ 13 - 14\n",
  "#   - theta = 30 deg --> e ~ 28 - 33\n",
  "# Applying a flat cut e <= 15 everywhere artificially discards valid reaction protons\n",
  "# hitting near the detector edges! Below we calculate theta(x, y) for every track\n",
  "# and apply a dynamic, position-dependent threshold e_max(theta).\n",
  "\n",
  "import numpy as np\n",
  "import matplotlib.pyplot as plt\n",
  "\n",
  "d0_cm = 6.0  # Distance from target to CR-39 plate\n",
  "\n",
  "# 1. Calculate radial offset and incident angle for raw tracks\n",
  "if 'data_raw' in locals() and hasattr(data_raw, 'tracks') and len(data_raw.tracks) > 0:\n",
  "    t_df = data_raw.tracks.copy()\n",
  "else:\n",
  "    t_df = tracks.copy()  # Fallback to current loaded tracks\n",
  "\n",
  "t_df['r_cm'] = np.sqrt(t_df['x']**2 + t_df['y']**2)\n",
  "t_df['theta_deg'] = np.degrees(np.arctan2(t_df['r_cm'], d0_cm))\n",
  "\n",
  "# 2. Dynamic threshold e_max(theta) model based on Fig 9b curve:\n",
  "# Baseline e_max = 15.0 at 0 deg, scaling quadratically up to 35.0 at 30 deg\n",
  "def e_max_threshold(theta_deg):\n",
  "    return 15.0 + 20.0 * (np.maximum(0.0, theta_deg) / 30.0)**2\n",
  "\n",
  "t_df['e_max_allowed'] = e_max_threshold(t_df['theta_deg'])\n",
  "t_df['pass_dynamic_e'] = t_df['e'] <= t_df['e_max_allowed']\n",
  "t_df['pass_flat_e']    = t_df['e'] <= 15.0\n",
  "\n",
  "# Summary counts\n",
  "n_flat = t_df['pass_flat_e'].sum()\n",
  "n_dyn  = t_df['pass_dynamic_e'].sum()\n",
  "print(f\"Incident Angle Analysis Across CR-39 Detector (d0 = {d0_cm} cm):\")\n",
  "print(f\"  Max Incident Angle on Plate: {t_df['theta_deg'].max():.2f} deg\")\n",
  "print(f\"  Mean Incident Angle        : {t_df['theta_deg'].mean():.2f} deg\")\n",
  "print(f\"  Tracks passing FLAT cut (e <= 15)       : {n_flat:,d}\")\n",
  "print(f\"  Tracks passing DYNAMIC cut e <= e_max(theta): {n_dyn:,d} (+{n_dyn - n_flat:,d} edge tracks recovered!)\")\n",
  "\n",
  "# 3. Plotting Incident Angle & Eccentricity Model\n",
  "fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)\n",
  "\n",
  "# Left: Eccentricity vs Incident Angle Scatter with Fig 9b Model Overlay\n",
  "ax1 = axes[0]\n",
  "thetas = np.linspace(0, 35, 100)\n",
  "ax1.plot(thetas, e_max_threshold(thetas), 'r--', lw=2.5, label=r'Dynamic Cut Boundary $e_{\\max}(\\theta)$ (Fig 9b)')\n",
  "ax1.axhline(15.0, color='blue', ls=':', lw=1.5, label='Flat Cut (e <= 15)')\n",
  "# Sample 5000 points for clear visualization\n",
  "sample_df = t_df.sample(n=min(5000, len(t_df)), random_state=42)\n",
  "ax1.scatter(sample_df['theta_deg'], sample_df['e'], c=sample_df['pass_dynamic_e'], \n",
  "            cmap='coolwarm', alpha=0.4, s=8, zorder=2)\n",
  "ax1.set_xlabel('Incident Angle $\\theta$ [degrees]', fontsize=11)\n",
  "ax1.set_ylabel('Track Eccentricity $e$', fontsize=11)\n",
  "ax1.set_title('Track Eccentricity vs Incident Angle $\\theta$', fontsize=12)\n",
  "ax1.legend(fontsize=9); ax1.grid(alpha=0.3)\n",
  "\n",
  "# Right: Spatial Map of Incident Angle theta(x, y) across detector\n",
  "ax2 = axes[1]\n",
  "sc = ax2.scatter(sample_df['x'], sample_df['y'], c=sample_df['theta_deg'], \n",
  "                 cmap='viridis', s=10, alpha=0.7)\n",
  "cbar = fig.colorbar(sc, ax=ax2)\n",
  "cbar.set_label('Incident Angle $\\theta$ [deg]')\n",
  "ax2.set_xlabel('Detector X [cm]'); ax2.set_ylabel('Detector Y [cm]')\n",
  "ax2.set_title('Spatial Distribution of Incident Angles $\\theta(x,y)$', fontsize=12)\n",
  "ax2.grid(alpha=0.3)\n",
  "\n",
  "fig.tight_layout()\n",
  "plt.show()\n"
 ]
}

# Insert the new cell at position 22 in notebook
nb['cells'].insert(22, new_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Inserted new Incident Angle & Dynamic Eccentricity cell into CR39 notebook!")
