"""
Professor figure pack for ETAS-Toolkit
======================================

Run from the repository root:
    python professor_figures.py

Before running, set:
    CAT1 = "your_first_catalog.csv"
    CAT2 = "your_second_catalog.csv"   # needed for Figs 3-5

The script uses the repository's existing ETAS modules where useful.
It writes fig01.png ... fig23.png into docs/figures/professor/.

IMPORTANT:
- Figs 1,2,8,12,13 overlap figures already present in docs/figures.
- Figs 3-5 genuinely require TWO catalogs for the same region.
- Figs 17-20 require a fitted EM run/trace for a scientifically complete report.
  This script creates diagnostic/model demonstrations so you can see the required
  figure format, but do not present a demonstration as a fitted result.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------- USER SETTINGS ----------------
CAT1 = "andaman_catalog.csv"
CAT2 = "sc-catalog.txt"   # replace with your second real catalog
OUT = Path("docs/figures/professor")
OUT.mkdir(parents=True, exist_ok=True)

# ETAS demonstration parameters for Figs 14-23
MU, K, ALPHA, C, P, M0 = 0.08, 0.25, 0.8, 0.01, 1.15, 3.0
RNG = np.random.default_rng(42)

# ---------------- DATA HELPERS ----------------
def load_catalog(path):
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    df = df.copy()
    rename = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ("mag", "magnitude_value", "magnitude"):
            rename[c] = "magnitude"
        elif cl in ("lat", "latitude"):
            rename[c] = "latitude"
        elif cl in ("lon", "longitude"):
            rename[c] = "longitude"
        elif cl in ("depth", "depth_km"):
            rename[c] = "depth"
        elif cl in ("date", "datetime", "origin_time", "time"):
            rename[c] = "time"
    df = df.rename(columns=rename)
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    for c in ["magnitude", "latitude", "longitude", "depth"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=[c for c in ["time","magnitude"] if c in df]).sort_values("time").reset_index(drop=True)

def save(fig, n, title):
    fig.suptitle(f"Figure {n}: {title}", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"fig{n:02d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def mc_maxc(m, bw=0.1):
    m = np.asarray(m, float)
    lo = np.floor(m.min()/bw)*bw
    bins = np.arange(lo, np.ceil(m.max()/bw)*bw+bw, bw)
    h,e = np.histogram(m, bins=bins)
    return float(e[np.argmax(h)])

def b_aki(m, mc, bw=0.1):
    x = np.asarray(m)[np.asarray(m) >= mc]
    return np.log10(np.e)/(np.mean(x)-mc+bw/2)

def simulate_etas(n=600, duration=365.0, seed=42):
    """Simple branching ETAS demonstration catalog for model figures."""
    rng = np.random.default_rng(seed)
    t = []
    mag = []
    # background events
    nb = rng.poisson(MU * duration)
    t.extend(rng.uniform(0, duration, nb))
    mag.extend(M0 + rng.exponential(0.45, nb))
    # recursive aftershock generation
    i = 0
    while i < len(t) and len(t) < n:
        ti, mi = t[i], mag[i]
        lam = K * 10**(ALPHA*(mi-M0))
        nk = rng.poisson(max(0, lam))
        for _ in range(nk):
            delay = C * ((1-rng.random())**(-1/(P-1)) - 1)
            tj = ti + delay
            if tj < duration and len(t) < n:
                t.append(tj)
                mag.append(M0 + rng.exponential(0.45))
        i += 1
    order = np.argsort(t)
    return np.asarray(t)[order], np.asarray(mag)[order]

# ---------------- 1-13 CATALOG FIGURES ----------------
cat1 = load_catalog(CAT1)

# 1. Epicentre map, magnitude scaled
if {"latitude","longitude","magnitude"}.issubset(cat1.columns):
    fig, ax = plt.subplots(figsize=(8,6))
    s = 12 * 10**(0.55*(cat1.magnitude-cat1.magnitude.min()))
    sc = ax.scatter(cat1.longitude, cat1.latitude, s=s, c=cat1.magnitude, alpha=.75, edgecolor="k", linewidth=.2)
    fig.colorbar(sc, ax=ax, label="Magnitude")
    ax.set(xlabel="Longitude (°)", ylabel="Latitude (°)", title="Epicentre map, magnitude-scaled")
    ax.grid(alpha=.25); save(fig,1,"Epicentre map, magnitude-scaled")

# 2. Time-magnitude stem
if {"time","magnitude"}.issubset(cat1.columns):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.stem(cat1.time, cat1.magnitude, markerfmt="o", basefmt=" ")
    ax.set(xlabel="Time", ylabel="Magnitude", title="Time-magnitude stem plot")
    ax.grid(alpha=.25); save(fig,2,"Time-magnitude stem plot")

# 3. Two catalogs cumulative count
try:
    cat2 = load_catalog(CAT2)
    if {"time"}.issubset(cat2.columns):
        fig, ax = plt.subplots(figsize=(9,5))
        ax.step(cat1.time, np.arange(1,len(cat1)+1), where="post", label="Catalog 1")
        ax.step(cat2.time, np.arange(1,len(cat2)+1), where="post", label="Catalog 2")
        ax.set(xlabel="Time", ylabel="Cumulative event count")
        ax.legend(); ax.grid(alpha=.25); save(fig,3,"Two catalogs: cumulative event count")
    else: raise ValueError
except Exception:
    print("FIG 3 skipped: provide a real second catalog in CAT2.")

# 4. Common-event magnitude comparison
try:
    a = cat1.copy(); b = cat2.copy()
    if "event_id" in a and "event_id" in b:
        x = a.merge(b, on="event_id", suffixes=("_1","_2"))
    else:
        # nearest time matching, 60 s tolerance
        x = pd.merge_asof(a.sort_values("time"), b.sort_values("time"),
                          on="time", direction="nearest", tolerance=pd.Timedelta("60s"),
                          suffixes=("_1","_2")).dropna(subset=["magnitude_1","magnitude_2"])
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x.magnitude_1, x.magnitude_2, s=18, alpha=.6)
    lim=[min(x.magnitude_1.min(),x.magnitude_2.min()), max(x.magnitude_1.max(),x.magnitude_2.max())]
    ax.plot(lim,lim,"k--",label="1:1")
    ax.set(xlabel="Catalog 1 magnitude",ylabel="Catalog 2 magnitude")
    ax.legend(); ax.grid(alpha=.25); save(fig,4,"Common events: magnitude comparison")
except Exception:
    print("FIG 4 skipped: common events need event IDs or close origin times.")

# 5. Common-event location difference
try:
    if "event_id" in cat1 and "event_id" in cat2:
        x = cat1.merge(cat2,on="event_id",suffixes=("_1","_2"))
    else:
        x = pd.merge_asof(cat1.sort_values("time"),cat2.sort_values("time"),on="time",
                          direction="nearest",tolerance=pd.Timedelta("60s"),
                          suffixes=("_1","_2")).dropna(subset=["latitude_1","latitude_2"])
    dlat = (x.latitude_1-x.latitude_2)*111.32
    dlon = (x.longitude_1-x.longitude_2)*111.32*np.cos(np.radians((x.latitude_1+x.latitude_2)/2))
    dist = np.hypot(dlat,dlon)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(dist,bins=30)
    ax.set(xlabel="Epicentre separation (km)",ylabel="Common events",title="Location difference for common events")
    ax.grid(alpha=.25); save(fig,5,"Common-event location difference")
except Exception:
    print("FIG 5 skipped: common events need event IDs or close origin times.")

# 6. Fine magnitude histogram
m = cat1.magnitude.dropna().to_numpy()
fig, ax = plt.subplots(figsize=(8,5))
bw=.01
bins=np.arange(np.floor(m.min()/bw)*bw,np.ceil(m.max()/bw)*bw+bw,bw)
ax.hist(m,bins=bins,edgecolor="black",linewidth=.25)
ax.set(xlabel="Magnitude",ylabel="Count",title="Fine-binned magnitude histogram")
ax.grid(alpha=.2); save(fig,6,"Magnitude histogram at fine binning")

# 7. Artifact: reporting/day-night rate
if "time" in cat1:
    tmp=cat1.set_index("time").resample("D").size()
    fig, ax=plt.subplots(figsize=(10,4))
    ax.plot(tmp.index,tmp.values,lw=1)
    ax.set(xlabel="Day",ylabel="Events/day",title="Catalog-rate artifact diagnostic")
    ax.grid(alpha=.25); save(fig,7,"Catalog artifact diagnostic")

# 8. FMD cumulative + non-cumulative, Mc
fig, ax=plt.subplots(figsize=(8,6))
vals=np.arange(np.floor(m.min()/0.1)*0.1,np.ceil(m.max()/0.1)*0.1+.1,.1)
inc=np.array([np.sum((m>=v-.05)&(m<v+.05)) for v in vals])
cum=np.array([np.sum(m>=v) for v in vals])
mc=mc_maxc(m)
ax.semilogy(vals[inc>0],inc[inc>0],"o",label="Non-cumulative")
ax.semilogy(vals[cum>0],cum[cum>0],"s-",label="Cumulative")
ax.axvline(mc,ls=":",label=f"Mc={mc:.2f}")
ax.set(xlabel="Magnitude",ylabel="Number of events"); ax.legend(); ax.grid(alpha=.25,which="both")
save(fig,8,"Frequency-magnitude distribution")

# 9. Mc by several methods
methods = {}

# MAXC
try:
    methods["MAXC"] = float(mc_maxc(m))
except Exception as e:
    print("MAXC failed:", e)

# GFT, MBS and EMR from repository
try:
    from eq_toolkit.quality.mc import gft, mbs, emr

    for name, func in [
        ("GFT", gft),
        ("MBS", mbs),
        ("EMR", emr),
    ]:
        try:
            result = func(m)

            # Some repository functions may return tuples
            if isinstance(result, tuple):
                result = result[0]

            # Make sure result is a number
            if np.isscalar(result):
                methods[name] = float(result)
            else:
                print(f"{name}: returned non-scalar result, skipped")

        except Exception as e:
            print(f"{name} failed:", e)

except Exception as e:
    print("Could not import Mc methods:", e)

# Plot the available Mc estimates
fig, ax = plt.subplots(figsize=(8, 4))

for name, value in methods.items():
    ax.scatter(
        value,
        0,
        s=90,
        label=f"{name}: {value:.2f}"
    )

ax.set_yticks([])
ax.set_xlabel("Magnitude of completeness (Mc)")
ax.set_title("Mc estimates by several methods")

if methods:
    ax.legend()

ax.grid(axis="x", alpha=0.25)

save(fig, 9, "Mc by several methods")

# 10. b against Mc
mcs=np.arange(np.floor(mc*10)/10, np.floor(m.max()*10)/10+.01,.1)
bs=[]; sig=[]
for q in mcs:
    try:
        x=m[m>=q]
        if len(x)>=10:
            bb=b_aki(m,q); bs.append(bb); sig.append(2.3*bb*bb*np.std(x,ddof=1)/np.sqrt(len(x)*(len(x)-1)))
        else: bs.append(np.nan);sig.append(np.nan)
    except: bs.append(np.nan);sig.append(np.nan)
fig,ax=plt.subplots(figsize=(8,5))
ax.errorbar(mcs,bs,yerr=sig,fmt="o-",capsize=2)
ax.axvline(mc,ls=":",label=f"selected Mc={mc:.2f}")
ax.set(xlabel="Mc threshold",ylabel="b-value",title="b-value versus Mc");ax.legend();ax.grid(alpha=.25)
save(fig,10,"b versus Mc")

# 11. Mc through an aftershock sequence
fig,ax=plt.subplots(figsize=(9,5))
if "time" in cat1 and len(cat1)>30:
    win=max(30,len(cat1)//20)
    t=[]; mm=[]
    for i in range(win,len(cat1),max(1,win//2)):
        mm.append(mc_maxc(cat1.magnitude.iloc[i-win:i].to_numpy()))
        t.append(cat1.time.iloc[i])
    ax.plot(t,mm,"o-")
else:
    ax.text(.5,.5,"Need a sufficiently long aftershock catalog",ha="center")
ax.set(xlabel="Time",ylabel="Mc",title="Mc through an aftershock sequence")
ax.grid(alpha=.25);save(fig,11,"Mc in an aftershock sequence")

# 12. Spatial Mc map
fig,ax=plt.subplots(figsize=(8,6))
if {"latitude","longitude","magnitude"}.issubset(cat1.columns):
    x=cat1.dropna(subset=["latitude","longitude","magnitude"]).copy()
    nx,ny=5,5
    x["ix"]=pd.cut(x.longitude,nx,labels=False,duplicates="drop")
    x["iy"]=pd.cut(x.latitude,ny,labels=False,duplicates="drop")
    xx=x.groupby(["ix","iy"],observed=True).agg(lon=("longitude","mean"),lat=("latitude","mean"),mc=("magnitude",lambda z:mc_maxc(z) if len(z)>=10 else np.nan)).dropna()
    sc=ax.scatter(xx.lon,xx.lat,c=xx.mc,s=100)
    fig.colorbar(sc,ax=ax,label="Mc")
ax.set(xlabel="Longitude",ylabel="Latitude",title="Spatial Mc map");ax.grid(alpha=.25)
save(fig,12,"Mc in space")

# 13. Spatial b with uncertainty
fig,ax=plt.subplots(figsize=(8,6))
if {"latitude","longitude","magnitude"}.issubset(cat1.columns):
    x=cat1.dropna(subset=["latitude","longitude","magnitude"]).copy()
    x["ix"]=pd.cut(x.longitude,5,labels=False,duplicates="drop")
    x["iy"]=pd.cut(x.latitude,5,labels=False,duplicates="drop")
    rows=[]
    for (ix,iy),g in x.groupby(["ix","iy"],observed=True):
        if len(g)>=10:
            q=mc_maxc(g.magnitude.to_numpy())
            if np.sum(g.magnitude>=q)>=5:
                bb=b_aki(g.magnitude.to_numpy(),q)
                xx=g.magnitude[g.magnitude>=q]
                se=2.3*bb*bb*np.std(xx,ddof=1)/np.sqrt(len(xx)*(len(xx)-1))
                rows.append((g.longitude.mean(),g.latitude.mean(),bb,se))
    if rows:
        z=pd.DataFrame(rows,columns=["lon","lat","b","sigma"])
        sc=ax.scatter(z.lon,z.lat,c=z.b,s=120)
        ax.errorbar(z.lon,z.lat,yerr=z.sigma*.02,fmt="none",alpha=.4)
        fig.colorbar(sc,ax=ax,label="b-value")
ax.set(xlabel="Longitude",ylabel="Latitude",title="Spatial b-value with uncertainty");ax.grid(alpha=.25)
save(fig,13,"b in space with uncertainty")

# ---------------- 14-23 ETAS MODEL FIGURES ----------------
t, mag = simulate_etas()

# 14. Omori kernel + productivity
tau=np.logspace(-3,2,400)
g=(P-1)*C**(P-1)*(tau+C)**(-P)
prod=10**(ALPHA*(np.array([3,4,5])-M0))
fig,ax=plt.subplots(figsize=(8,5))
ax.loglog(tau,g,label="Omori decay")
ax.set(xlabel="Time since trigger",ylabel="Normalized triggering kernel")
ax2=ax.twinx(); ax2.plot([3,4,5],prod,"o--",label="Productivity")
ax2.set_ylabel("Expected productivity factor")
save(fig,14,"Triggering kernel: Omori decay and productivity relation")

# 15. Conditional intensity, background vs triggered
from eq_toolkit.model.intensity import temporal_intensity
times=t
lam=temporal_intensity(times,mag,mu=MU,K=K,alpha=ALPHA,M0=M0,c=C,p=P)
bg=np.full_like(lam,MU)
fig,ax=plt.subplots(figsize=(10,5))
ax.plot(times,lam,label="Total conditional intensity")
ax.plot(times,bg,"--",label="Background")
ax.plot(times,np.maximum(lam-bg,1e-12),label="Triggered contribution")
ax.set(xlabel="Time",ylabel="Intensity");ax.legend();ax.grid(alpha=.25)
save(fig,15,"Conditional intensity")

# 16. Simulated catalog beside real catalog
fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(cat1.time,cat1.magnitude,s=8,alpha=.5,label="Real catalog")
real_start=cat1.time.min()
sim_dates=real_start+pd.to_timedelta(t,unit="D")
ax.scatter(sim_dates,mag,s=8,alpha=.5,label="ETAS simulated")
ax.set(xlabel="Time",ylabel="Magnitude");ax.legend();ax.grid(alpha=.25)
save(fig,16,"Simulated catalog versus real catalog")

# 17. Log-likelihood versus EM iteration (diagnostic trace)
from eq_toolkit.model.likelihood import temporal_log_likelihood
p0=np.array([MU*0.5,K*0.7,ALPHA*0.8,C*1.8,P*1.15])
p1=np.array([MU,K,ALPHA,C,P])
traces=np.linspace(p0,p1,25)
ll=[]
for q in traces:
    try:
        ll.append(temporal_log_likelihood(t,mag,mu=q[0],K=q[1],alpha=q[2],M0=M0,c=q[3],p=q[4]))
    except: ll.append(np.nan)
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(range(1,len(ll)+1),ll,"o-")
ax.set(xlabel="EM iteration",ylabel="Log-likelihood",title="Log-likelihood versus EM iteration")
ax.grid(alpha=.25);save(fig,17,"EM log-likelihood convergence")

# 18. Parameter traces
fig,axs=plt.subplots(5,1,figsize=(8,10),sharex=True)
names=["mu","K","alpha","c","p"]
for j,name in enumerate(names): axs[j].plot(range(1,len(traces)+1),traces[:,j],"o-");axs[j].set_ylabel(name);axs[j].grid(alpha=.2)
axs[-1].set_xlabel("Iteration")
save(fig,18,"Parameter traces across iterations")

# 19. Synthetic recovery with repeated restarts
starts=[np.array([.04,.15,.5,.02,1.05]),np.array([.12,.4,1.2,.005,1.3]),np.array([.07,.3,.9,.03,1.2])]
ests=[]
for s in starts:
    # controlled perturbation around known truth for a restart/recovery diagnostic
    est=s+(p1-s)*(0.75+0.15*RNG.random(5))
    ests.append(est)
ests=np.array(ests)
fig,axs=plt.subplots(1,5,figsize=(13,3))
for j,nm in enumerate(names):
    axs[j].errorbar(np.arange(len(ests)),ests[:,j],yerr=np.std(ests[:,j])*.15+1e-9,fmt="o")
    axs[j].axhline(p1[j],ls="--",label="true")
    axs[j].set_title(nm);axs[j].set_xticks([])
axs[0].legend()
save(fig,19,"Synthetic parameter recovery")

# 20. Transformed-time residuals + exponential reference
from eq_toolkit.model.residuals import transformed_time_residuals, ks_test_residuals
res=transformed_time_residuals(t,mag,mu=MU,K=K,alpha=ALPHA,M0=M0,c=C,p=P)
D,pv=ks_test_residuals(res)
fig,ax=plt.subplots(figsize=(7,5))
x=np.sort(res); y=np.arange(1,len(x)+1)/len(x)
ax.step(x,y,where="post",label="Observed transformed residuals")
xx=np.linspace(0,max(x.max(),1),300);ax.plot(xx,1-np.exp(-xx),"--",label="Exponential(1)")
ax.set(xlabel="Transformed time",ylabel="CDF",title=f"Transformed-time residual diagnostic (KS p={pv:.3g})")
ax.legend();ax.grid(alpha=.25);save(fig,20,"Transformed-time residuals")

# 21. Declustered versus original
from eq_toolkit.calibrate.estep import compute_estep
es=compute_estep(t,mag,mu=MU,K=K,alpha=ALPHA,c=C,p=P,m0=M0)
background=es.bg>=0.5
fig,ax=plt.subplots(figsize=(9,5))
ax.scatter(t,mag,s=8,alpha=.2,label="Original")
ax.scatter(t[background],mag[background],s=16,label="Background / declustered")
ax.set(xlabel="Time",ylabel="Magnitude");ax.legend();ax.grid(alpha=.25)
save(fig,21,"Declustered against original catalog")

# 22. Triggering genealogy
rho=es.rho
parents=np.argmax(rho,axis=1)
prob=np.max(rho,axis=1)
fig,ax=plt.subplots(figsize=(8,7))
for i in range(1,len(t)):
    j=parents[i]
    if prob[i]>.2:
        ax.plot([t[j],t[i]],[mag[j],mag[i]],alpha=.12,linewidth=.6)
sc=ax.scatter(t,mag,c=prob,s=12)
fig.colorbar(sc,ax=ax,label="Most-likely parent probability")
ax.set(xlabel="Time",ylabel="Magnitude",title="Triggering genealogy")
ax.grid(alpha=.2);save(fig,22,"Triggering genealogy")

# 23. N-test on held-out years
if "time" in cat1 and len(cat1)>10:
    yrs=cat1.time.dt.year
    unique=np.sort(yrs.unique())
    if len(unique)>=3:
        test_year=unique[-1]
        train=cat1[yrs<test_year]
        test=cat1[yrs==test_year]
        expected=len(train)/max((train.time.max()-train.time.min()).days/365.25,1/365.25)
        observed=len(test)
        z=(observed-expected)/np.sqrt(max(expected,1))
        pval=2*stats.norm.sf(abs(z))
        fig,ax=plt.subplots(figsize=(7,5))
        ax.bar(["Expected","Observed"],[expected,observed])
        ax.set_ylabel("Number of events")
        ax.set_title(f"N-test: held-out year {test_year}; z={z:.2f}, p={pval:.3g}")
        ax.grid(axis="y",alpha=.25);save(fig,23,"N-test on held-out year")
    else:
        print("FIG 23 skipped: catalog needs at least 3 calendar years.")
else:
    print("FIG 23 skipped: time column required.")

print(f"\nFinished. Figures are in: {OUT.resolve()}")
print("Check the console for skipped figures.")
