"""
Five CR-39 diagnostics, run on the existing scan. No new data needed.

  1. Radial eccentricity gradient  -- is there ANY directional signal in there?
  2. The unused average-contrast (a) field -- does (c, a) separate populations?
  3. Frame-level quality cuts      -- focus tilt and dust/scratch frames
  4. Nearest-neighbour declustering -- Poisson singles vs correlated junk
  5. Net signal vs c_max sweep      -- plateau (real) or scaling (noise)

Output: CR-39_Python_code/plots/diag_1..5_*.png  plus a printed report.
"""
import os, struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CPSA = os.path.join(ROOT, "CR-39_data",
                    "A20260413_13CD_125-50umTaFilter_W75_5hr_40x_backside_s0.cpsa")
PLOTS = os.path.join(ROOT, "CR-39_Python_code", "plots")
os.makedirs(PLOTS, exist_ok=True)

# geometry / regions, matching the notebook
R_CR39 = 6.0
SIG = dict(xmin=-1.5, xmax=1.5, ymin=-1.8, ymax=1.4)
BG = dict(xmin=-1.37, xmax=1.47, ymin=1.50, ymax=1.89)
A_SIG = (SIG["xmax"]-SIG["xmin"]) * (SIG["ymax"]-SIG["ymin"])
A_BG = (BG["xmax"]-BG["xmin"]) * (BG["ymax"]-BG["ymin"])
SCALE = A_SIG / A_BG

plt.rcParams.update({"font.size": 11, "figure.dpi": 140,
                     "xtick.direction": "in", "ytick.direction": "in"})


# ─────────────────────────────────────────────────────────── reader
def read_cpsa(path):
    with open(path, "rb") as f:
        ri = lambda: struct.unpack("<i", f.read(4))[0]
        rf = lambda: struct.unpack("<f", f.read(4))[0]
        h = dict(version=ri(), nx=ri(), ny=ri(), nbins=ri(),
                 pixel_size=1e-4*rf(), ppb=rf(), border=ri(),
                 clim=ri(), elim=ri(), M=ri(), fw=ri(), fh=ri())
        ps = h["pixel_size"]
        fw, fh = h["fw"]*ps, h["fh"]*ps
        nfr = h["nx"] * h["ny"]
        frames, chunks = [], []
        for _ in range(nfr):
            num = ri(); xp = 1e-5*ri(); yp = 1e-5*ri(); nt = ri()
            f.read(12)
            foc = 1e-2*ri(); xi = ri(); yi = ri()
            frames.append((num, xp, yp, nt, foc, xi, yi))
            if nt == 0:
                continue
            d = np.frombuffer(f.read(2*nt), dtype="<i2")
            e = np.frombuffer(f.read(nt), dtype="<i1")
            c = np.frombuffer(f.read(nt), dtype="<i1")
            a = np.frombuffer(f.read(nt), dtype="<i1")
            xr = np.frombuffer(f.read(2*nt), dtype="<i2")
            yr = np.frombuffer(f.read(2*nt), dtype="<i2")
            ch = np.empty((nt, 8))
            ch[:, 0] = num
            ch[:, 1] = 100.0*d*ps
            ch[:, 2] = xp - 0.5*fw + xr*ps
            ch[:, 3] = yp - 0.5*fh + yr*ps
            ch[:, 4] = e; ch[:, 5] = c; ch[:, 6] = a; ch[:, 7] = foc
            chunks.append(ch)
    arr = np.vstack(chunks)
    fr = np.array(frames, dtype=float)
    return h, fr, dict(frame=arr[:, 0], d=arr[:, 1], x=arr[:, 2], y=arr[:, 3],
                       e=arr[:, 4], c=arr[:, 5], a=arr[:, 6], foc=arr[:, 7])


print("reading", os.path.basename(CPSA))
H, FR, T = read_cpsa(CPSA)
on = (np.abs(T["x"]) <= 2.5) & (np.abs(T["y"]) <= 2.5)
for k in T:
    T[k] = T[k][on]
N = len(T["d"])
T["r"] = np.hypot(T["x"], T["y"])
print(f"  {N:,} tracks on the piece;  e range {T['e'].min():.0f}-{T['e'].max():.0f}, "
      f"c range {T['c'].min():.0f}-{T['c'].max():.0f}, a range {T['a'].min():.0f}-{T['a'].max():.0f}")

inbox = ((T["x"] >= SIG["xmin"]) & (T["x"] <= SIG["xmax"]) &
         (T["y"] >= SIG["ymin"]) & (T["y"] <= SIG["ymax"]))
innotch = ((T["x"] >= BG["xmin"]) & (T["x"] <= BG["xmax"]) &
           (T["y"] >= BG["ymin"]) & (T["y"] <= BG["ymax"]))
qual = (T["d"] >= 2.0) & (T["d"] <= 13.0) & (T["e"] <= 15) & (T["c"] <= 20)

SEP = "=" * 74


# ══════════════════════════════════════════════ 1. radial eccentricity gradient
print("\n" + SEP + "\n  DIAG 1 — RADIAL ECCENTRICITY GRADIENT\n" + SEP)
print("  Signal protons come from a point source: a track at radius r arrived")
print("  at theta = arctan(r/6.0). Eccentricity must RISE with r.")
print("  Neutron recoils are born isotropically in the bulk: FLAT with r.\n")

redges = np.arange(0.0, 2.51, 0.25)
rmid = 0.5*(redges[:-1] + redges[1:])
theta = np.degrees(np.arctan(rmid / R_CR39))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
res1 = {}
for ax, (lbl, msk) in zip(axes, [("all tracks", np.ones(N, bool)),
                                 ("quality-cut tracks", qual)]):
    me, se, nn = [], [], []
    for i in range(len(rmid)):
        m = msk & (T["r"] >= redges[i]) & (T["r"] < redges[i+1])
        k = m.sum()
        nn.append(k)
        me.append(T["e"][m].mean() if k else np.nan)
        se.append(T["e"][m].std()/np.sqrt(k) if k > 1 else np.nan)
    me, se = np.array(me), np.array(se)
    res1[lbl] = (me, se, np.array(nn))
    ax.errorbar(rmid, me, yerr=se, fmt="o-", ms=6, capsize=3, color="black")
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(rmid[::2])
    ax2.set_xticklabels([f"{t:.0f}" for t in theta[::2]], fontsize=9)
    ax2.set_xlabel("implied incident angle $\\theta$ (deg)", fontsize=9)
    ax.set_xlabel("radius on plate r (cm)")
    ax.set_ylabel("mean eccentricity")
    ax.set_title(f"{lbl}  (N = {int(msk.sum()):,})", fontsize=11)
    ax.grid(alpha=0.3)
fig.suptitle("Diag 1 — does eccentricity rise with radius? (point source vs isotropic)", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "diag_1_radial_eccentricity.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

for lbl, (me, se, nn) in res1.items():
    ok = ~np.isnan(me) & (nn > 100)
    A = np.vstack([np.ones(ok.sum()), rmid[ok]]).T
    w = 1/se[ok]**2
    cov = np.linalg.inv(A.T @ np.diag(w) @ A)
    beta = cov @ (A.T @ np.diag(w) @ me[ok])
    sig = beta[1]/np.sqrt(cov[1, 1])
    print(f"  {lbl:20s}  e(centre)={me[ok][0]:5.2f}  e(edge)={me[ok][-1]:5.2f}"
          f"   slope = {beta[1]:+.3f}/cm  ({sig:+.1f} sigma)")
    res1[lbl] = (me, se, nn, sig)
print("\n  Przybocki 5 h etch, 2-3 MeV protons: e ~ 4 at 0 deg -> ~13-20 at 20 deg.")
print("  Our plate only spans 0 to 22.6 deg at r = 6.0 cm.")


# ══════════════════════════════════════════════ 2. the unused `a` field
print("\n" + SEP + "\n  DIAG 2 — THE UNUSED AVERAGE-CONTRAST FIELD\n" + SEP)
ce = np.arange(0, 101, 2.0)
ae = np.arange(0, 101, 2.0)
Hs, _, _ = np.histogram2d(T["c"][inbox], T["a"][inbox], bins=[ce, ae])
Hb, _, _ = np.histogram2d(T["c"][innotch], T["a"][innotch], bins=[ce, ae])
Hd = Hs - Hb*SCALE

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ext = [ce[0], ce[-1], ae[0], ae[-1]]
for ax, M, ttl, diff in [(axes[0], Hs, f"signal box   N={Hs.sum():,.0f}", False),
                         (axes[1], Hb*SCALE, f"notch x{SCALE:.2f}   N={(Hb*SCALE).sum():,.0f}", False),
                         (axes[2], Hd, f"difference   N={Hd.sum():,.0f}", True)]:
    if diff:
        L = np.abs(M).max()
        im = ax.imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                       norm=SymLogNorm(linthresh=1.0, vmin=-L, vmax=L))
    else:
        im = ax.imshow(M.T, origin="lower", extent=ext, aspect="auto", cmap="inferno",
                       norm=LogNorm(vmin=1))
    fig.colorbar(im, ax=ax, pad=0.02)
    ax.axvline(20, color="lime", ls="--", lw=1.4)
    ax.set_xlabel("normal contrast c"); ax.set_title(ttl, fontsize=10)
axes[0].set_ylabel("average contrast a")
fig.suptitle("Diag 2 — (c, a) plane: does the second contrast field separate anything?", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "diag_2_contrast_plane.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

corr = np.corrcoef(T["c"], T["a"])[0, 1]
print(f"  Pearson correlation between c and a : {corr:+.4f}")
print(f"  a range {T['a'].min():.0f}-{T['a'].max():.0f}, median {np.median(T['a']):.0f}")
pos = Hd[Hd > 0].sum(); neg = Hd[Hd < 0].sum()
print(f"  difference map: +{pos:,.0f} / {neg:,.0f}  net {Hd.sum():,.0f}")
if corr > 0.9:
    print("  -> c and a are nearly the same measurement; little independent power.")
else:
    print("  -> c and a carry partly independent information; worth a 2D cut.")


# ══════════════════════════════════════════════ 3. frame-level quality
print("\n" + SEP + "\n  DIAG 3 — FRAME-LEVEL QUALITY CUTS\n" + SEP)
fnum, fx, fy, fnt, ffoc = FR[:, 0], FR[:, 1], FR[:, 2], FR[:, 3], FR[:, 4]
live = fnt > 0
print(f"  frames total {len(FR):,},  with tracks {live.sum():,}")
print(f"  focus range {ffoc.min():.2f} to {ffoc.max():.2f}")
print(f"  tracks/frame: mean {fnt[live].mean():.1f}, median {np.median(fnt[live]):.0f}, "
      f"max {fnt.max():.0f}")

q75, q25 = np.percentile(fnt[live], [75, 25])
thr = q75 + 3*(q75-q25)
hot = live & (fnt > thr)
hot_tracks = fnt[hot].sum()
print(f"  'hot' frame threshold (Q3 + 3*IQR) : {thr:.0f} tracks/frame")
print(f"  hot frames : {hot.sum():,}  holding {hot_tracks:,.0f} tracks "
      f"({hot_tracks/fnt.sum()*100:.2f}% of all)")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
axes[0].hist(ffoc[live], bins=80, color="steelblue")
axes[0].set_xlabel("frame focus"); axes[0].set_ylabel("frames"); axes[0].set_title("focus distribution")
sc = axes[1].scatter(fx[live], fy[live], c=ffoc[live], s=2, cmap="viridis")
fig.colorbar(sc, ax=axes[1], label="focus")
axes[1].set_xlabel("x (cm)"); axes[1].set_ylabel("y (cm)"); axes[1].set_title("focus vs position (tilt)")
axes[2].hist(fnt[live], bins=np.arange(0, np.percentile(fnt[live], 99.9)+2), color="darkorange")
axes[2].axvline(thr, color="red", ls="--", label=f"hot cut = {thr:.0f}")
axes[2].set_yscale("log"); axes[2].set_xlabel("tracks per frame"); axes[2].set_ylabel("frames")
axes[2].legend(); axes[2].set_title("tracks/frame (dust clusters in the tail)")
fig.suptitle("Diag 3 — frame-level quality", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "diag_3_frame_quality.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# does focus correlate with measured track properties?
from scipy.stats import spearmanr
fmap = {int(n): f for n, f in zip(fnum, ffoc)}
tf = T["foc"]
print(f"  Spearman focus vs diameter   : {spearmanr(tf, T['d']).statistic:+.4f}")
print(f"  Spearman focus vs contrast   : {spearmanr(tf, T['c']).statistic:+.4f}")
print(f"  Spearman focus vs eccentricity: {spearmanr(tf, T['e']).statistic:+.4f}")


# ══════════════════════════════════════════════ 4. nearest-neighbour
print("\n" + SEP + "\n  DIAG 4 — NEAREST-NEIGHBOUR DECLUSTERING\n" + SEP)
from scipy.spatial import cKDTree
sel = qual
pts = np.column_stack([T["x"][sel], T["y"][sel]])
tree = cKDTree(pts)
dnn, _ = tree.query(pts, k=2)
dnn = dnn[:, 1]
area = np.pi*2.5**2
dens = len(pts)/area
# 2D Poisson NN distance: P(<s) = 1 - exp(-pi*n*s^2), median = sqrt(ln2/(pi*n))
med_poisson = np.sqrt(np.log(2)/(np.pi*dens))
med_obs = np.median(dnn)
print(f"  quality tracks {len(pts):,},  density {dens:,.0f} /cm^2")
print(f"  median NN distance, observed : {med_obs*1e4:8.2f} um")
print(f"  median NN distance, Poisson  : {med_poisson*1e4:8.2f} um")
print(f"  ratio obs/Poisson            : {med_obs/med_poisson:8.3f}")

s = np.linspace(0, np.percentile(dnn, 99), 300)
cdf_obs = np.searchsorted(np.sort(dnn), s)/len(dnn)
cdf_poi = 1 - np.exp(-np.pi*dens*s**2)
excess = cdf_obs - cdf_poi
i_pk = int(np.argmax(excess))
s_cut = s[i_pk]
nclust = int((dnn < s_cut).sum() - len(dnn)*cdf_poi[i_pk])
print(f"  largest excess over Poisson at s = {s_cut*1e4:.1f} um "
      f"(+{excess[i_pk]*100:.2f} percentage points)")
print(f"  implied clustered excess     : {max(nclust,0):,} tracks "
      f"({max(nclust,0)/len(pts)*100:.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
axes[0].hist(dnn*1e4, bins=200, density=True, color="steelblue", alpha=.75, label="observed")
sd = np.linspace(0, np.percentile(dnn, 99.5), 400)
axes[0].plot(sd*1e4, 2*np.pi*dens*sd*np.exp(-np.pi*dens*sd**2)/1e4, "r-", lw=2,
             label="Poisson (random)")
axes[0].set_xlabel("nearest-neighbour distance (um)"); axes[0].set_ylabel("pdf")
axes[0].legend(); axes[0].set_title("NN distance: observed vs random")
axes[1].plot(s*1e4, excess*100, "k-", lw=1.8)
axes[1].axhline(0, color="gray", lw=1)
axes[1].axvline(s_cut*1e4, color="red", ls="--", label=f"max excess @ {s_cut*1e4:.0f} um")
axes[1].set_xlabel("distance s (um)"); axes[1].set_ylabel("CDF(obs) - CDF(Poisson)  [pp]")
axes[1].legend(); axes[1].set_title("clustering excess")
fig.suptitle("Diag 4 — are the tracks spatially random?", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "diag_4_nearest_neighbour.png"), dpi=200, bbox_inches="tight")
plt.close(fig)


# ══════════════════════════════════════════════ 5. net signal vs c_max
print("\n" + SEP + "\n  DIAG 5 — NET SIGNAL vs CONTRAST CUT\n" + SEP)
print("  Real signal saturates as the cut opens. Pure noise keeps scaling.\n")
base = (T["d"] >= 2.0) & (T["d"] <= 13.0) & (T["e"] <= 15)
cmaxes = np.arange(5, 51, 1.0)
nets, errs, boxes, notches = [], [], [], []
for cm in cmaxes:
    k = base & (T["c"] <= cm)
    nb = int((k & inbox).sum()); nn_ = int((k & innotch).sum())
    boxes.append(nb); notches.append(nn_)
    nets.append(nb - nn_*SCALE)
    errs.append(np.sqrt(nb + SCALE**2*nn_))
nets, errs = np.array(nets), np.array(errs)
boxes, notches = np.array(boxes), np.array(notches)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(cmaxes, boxes, "-", color="crimson", lw=2, label="signal box (raw)")
axes[0].plot(cmaxes, notches*SCALE, "-", color="dodgerblue", lw=2, label=f"notch x{SCALE:.2f}")
axes[0].set_xlabel("contrast cut  c $\\leq$ c$_{max}$"); axes[0].set_ylabel("tracks")
axes[0].legend(); axes[0].grid(alpha=.3); axes[0].set_title("raw counts vs cut")
axes[1].errorbar(cmaxes, nets, yerr=errs, fmt="o-", ms=3, color="black",
                 ecolor="gray", capsize=2)
axes[1].axvline(20, color="lime", ls="--", lw=2, label="our cut c $\\leq$ 20")
axes[1].axhline(1451, color="purple", ls=":", lw=2, label="SBD predicts ~1,451")
axes[1].set_xlabel("contrast cut  c $\\leq$ c$_{max}$")
axes[1].set_ylabel("net signal (box - scaled notch)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=.3); axes[1].set_title("net signal vs cut")
fig.suptitle("Diag 5 — plateau (real) or monotonic growth (noise)?", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "diag_5_contrast_sweep.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"  {'c_max':>6}{'box':>10}{'notch x8.67':>14}{'net':>11}{'+/-':>9}{'net/box':>10}")
for cm in (5, 10, 15, 20, 30, 40, 50):
    i = int(np.where(cmaxes == cm)[0][0])
    print(f"  {cm:>6.0f}{boxes[i]:>10,}{notches[i]*SCALE:>14,.0f}"
          f"{nets[i]:>11,.0f}{errs[i]:>9,.0f}{nets[i]/max(boxes[i],1)*100:>9.1f}%")
g = nets[cmaxes == 50][0] / nets[cmaxes == 20][0]
print(f"\n  net(c<=50) / net(c<=20) = {g:.2f}x  "
      f"(box raw grows {boxes[cmaxes==50][0]/boxes[cmaxes==20][0]:.2f}x over the same range)")
print("\nplots written to", os.path.relpath(PLOTS, ROOT))
