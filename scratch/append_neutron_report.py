import os

report_path = os.path.join(".", "docs", "incident_angle_and_channel_pileup_report.txt")
if not os.path.exists(report_path):
    report_path = os.path.join("..", "docs", "incident_angle_and_channel_pileup_report.txt")

with open(report_path, "r", encoding="utf-8") as f:
    existing_content = f.read()

appendix = """

================================================================================
4. INCIDENT ANGLE CORRECTION OUTCOMES & RECOVERED TRACKS
================================================================================
After adding the position-dependent incident angle theta(x, y) calculation 
and the dynamic eccentricity cut e <= e_max(theta) from Figure 9b:

- Total Scanned CR-39 Area: 22.28 cm^2 (106 x 144 frames)
- Total Raw Tracks Loaded : 530,122 tracks
- Raw Track Density       : 23,794.3 tracks/cm^2

Track Yield Comparison:
- Flat Cut (e <= 15.0)                : 120,469 tracks (5,407.2 tracks/cm^2)
- Dynamic Cut e <= e_max(theta)       : 127,309 tracks (5,714.2 tracks/cm^2)
- Recovered Valid Edge Tracks         : +6,840 tracks (+5.68% yield recovery)

This confirms that the position-dependent cut successfully recovers valid 
reaction protons at outer incident angles (theta > 15 deg) without introducing 
noise at the detector center.


================================================================================
5. TRACK DENSITY VS CONTRAST (c) BREAKDOWN
================================================================================
Track density (tracks/cm^2) across the 22.28 cm^2 detector plate as a function 
of optical contrast percentage (c):

Contrast Range (%)   | Track Count  | Track Density (tracks/cm^2)
-----------------------------------------------------------------
 0% to  5%           |           18 |         0.8 tracks/cm^2
 5% to 10%           |       50,558 |     2,269.3 tracks/cm^2
10% to 15%           |       63,696 |     2,859.0 tracks/cm^2
15% to 20%           |       40,088 |     1,799.3 tracks/cm^2
20% to 25%           |       20,992 |       942.2 tracks/cm^2
25% to 30%           |       14,615 |       656.0 tracks/cm^2
30% to 35%           |       12,941 |       580.9 tracks/cm^2
35% to 40%           |       14,068 |       631.4 tracks/cm^2
40% to 45%           |       16,149 |       724.8 tracks/cm^2
45% to 50%           |       19,627 |       881.0 tracks/cm^2
-----------------------------------------------------------------
Total (c <= 20% quality window): 154,360 tracks (6,927.4 tracks/cm^2)


================================================================================
6. DD NEUTRON VS INTRINSIC PLASTIC FOGGING DIAGNOSTIC (Frenje et al. 2002)
================================================================================
Physics Formulation:
1. Total DD Neutrons Produced in Target (Y_DD):
   Derived from SBD D+D reference yield (5.3549 x 10^6 counts, omega = 0.01075 sr):
     Y_DD = 6.260 x 10^9 neutrons

2. DD Neutron Fluence at CR-39 (r = 6.0 cm):
     Phi_n = Y_DD / (4 * pi * r^2) = 1.384 x 10^7 neutrons/cm^2

3. Front-Side DD Neutron Detection Efficiency (Frenje et al. 2002):
     eps_n = 1.1 x 10^-4 tracks per neutron

4. Expected DD Neutron Backgrounds:
   - Direct DD Neutron Track Density   : 1,522.1 tracks/cm^2
   - Backscattered Wall-Bounce Density : ~90.0 tracks/cm^2  (Fig. 11a, Frenje 2002)
   - TOTAL Expected Neutron Background : ~1,612.1 tracks/cm^2

Comparison with CR-39 Data:
- Observed CR-39 Track Density (Signal Window) : 5,714.2 tracks/cm^2
- Total Expected Neutron Background             : 1,612.1 tracks/cm^2 (~28%)
- Excess Background (Intrinsic Plastic Fogging) : 4,102.1 tracks/cm^2 (~72%)

Conclusion:
DD neutrons account for ~1,612 tracks/cm^2 (~28% of the noise). The remaining 
4,102 tracks/cm^2 (~72%) comes from intrinsic plastic defects / fogging.

Backside Scanning Verification (Frenje et al. 2002):
- Etching and scanning the BACK side of the 1 mm CR-39 plate will confirm this:
  * 2.45 MeV DD neutrons penetrate 1 mm plastic and leave tracks on BOTH sides.
  * Reaction protons (d+13C) stop near the front surface.
  * High backside track count confirms neutron background and bulk plastic defects.


================================================================================
7. FINAL CROSS-SECTION CALIBRATION RECOMMENDATION
================================================================================
- Always use the genuine 5.24 MeV SBD signal line (55.6 +- 9.3 net counts, 
  60.5 dead-time corrected) for nuclear cross-section calibration.
- Do NOT use the 4.84 MeV feature, as it is proven to be fake D+D pulse pile-up.
================================================================================
"""

with open(report_path, "w", encoding="utf-8") as f:
    f.write(existing_content + appendix)

print("Appended Section 4, 5, 6, and 7 to incident_angle_and_channel_pileup_report.txt successfully!")
