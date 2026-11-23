"""Compare a finished PMI network with the measured affinities.

    python analyze_networks.py val      # or exp, or both (the default)

Reads out_<net>_{free,complex}/, pools the production replicas and prints, per state, the MSLD
ddG_bind against the measured one. run_networks.sh calls this after both legs of a network finish.

The replicas are fed to the published ALF estimator as independent runs, which is what it is written for:
the value is the exponential average over replicas and the error bar is a bootstrap over them. That is a
real error estimate, unlike a bootstrap over chunks of one trajectory, whose chunks are correlated.

Measured values are same-study SPR Kd from the user's table (MDM2(25-109), 10 mM HEPES/150 mM NaCl pH 7.4):
  L2021  Li et al., Acta Pharm Sin B 2021, doi:10.1016/j.apsb.2021.06.010   (PMI reference Kd 2.7 nM)
  L2010  Li et al., J Mol Biol 2010,     doi:10.1016/j.jmb.2010.03.005     (PMI reference Kd 3.2 nM)
Only PMI-N8A comes from the 2010 study. The two studies agree on PMI-L10A (890 vs 897 nM), which is why
mixing them is defensible, but a cross-study row is marked and should be trusted less.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KCAL = 4.184
RT = 0.001987204258640832 * 298.15      # kcal/mol at the temperature the Kd values are reported at

# state label as the estimator writes it -> (sequence, Kd nM, Kd SD nM, study, name)
MEASURED = {
    "val": {
        "WT+WT":    ("TSFAEYWALLSP", 0.49, 0.21, "L2010", "PMI-N8A"),
        "ASN+WT":   ("TSFAEYWNLLSP", 2.7, 0.5, "L2021", "PMI"),
        "ASN+ALA":  ("TSFAEYWNLASP", 897, 33, "L2021", "PMI-L10A"),
        "WT+ALA":   ("TSFAEYWALASP", 229, 11, "L2021", "PMI-L10A+N8A"),
        "LEU+ALA":  ("TSFAEYWLLASP", 344, 8, "L2021", "PMI-L10A+N8L"),
        "THR+ALA":  ("TSFAEYWTLASP", 962, 29, "L2021", "PMI-L10A+N8T"),
    },
    "exp": {
        "WT+WT+WT":  ("TSFAEYWALLSP", 0.49, 0.21, "L2010", "PMI-N8A"),
        "ASN+WT+WT": ("TSFAEYWNLLSP", 2.7, 0.5, "L2021", "PMI"),
    },
}
# Network E sits on the L10-wild background, where only those two states were measured. The single
# increments it is built from (L9Q, L9M, S11M) were measured on the PMI-L10A background instead, so they
# are not rows here; they are what the additivity check below is compared against.
PMI_L10A_INCREMENTS = {       # kcal/mol vs PMI-L10A (897 nM), same study, for reference only
    "N8A": -0.81, "L9Q": -0.31, "L9M": -0.29, "S11M": -0.90, "P12Q": -1.14,
}
# The state each network's experimental column is referenced to, and what the MSLD run calls it.
REFERENCE = {"val": "ASN+WT", "exp": "ASN+WT+WT"}
# The states whose sum should equal the all-mutant corner if the mutations were additive.
# N8A + L9Q + S11M against the corner that carries all three: TSFAEYWAQLMP, PMI-M3's 8-11 motif.
ADDITIVITY = {"exp": (["WT+WT+WT", "ASN+GLN+WT", "ASN+WT+MET"], "WT+GLN+MET")}


def combine_replicas(G, kT):
    """Exponential average of per-replica free energies, per state, on the absolute scale.

    The published estimator ends by subtracting its first combination, which makes every state NaN as
    soon as that one combination is unsampled in a leg -- one missing state destroys the other eleven.
    Nothing here subtracts anything: states keep their own value, an unsampled one stays infinite, and
    the caller picks the reference it wants.
    """
    out = np.full(G.shape[1], np.inf)
    good = np.isfinite(G).any(axis=0)
    if good.any():
        g = G[:, good]
        gmin = np.nanmin(np.where(np.isfinite(g), g, np.inf), axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            out[good] = gmin - kT * np.log(np.mean(np.exp(-(g - gmin) / kT), axis=0))
    return out


def leg_estimate(workdir):
    """A leg's production replicas: (labels, dG kcal/mol, bootstrap error, endpoint counts, n replicas).

    dG is on the leg's own absolute scale; analyze() references it. An unsampled state is +inf.
    """
    from openmm_ld import alf_published
    from openmm_ld.prepared import load_prepared

    prepared = load_prepared(os.path.join(workdir, "prepared"))
    nsubs = [len(site.block_names) for site in prepared.mapping.sites]
    with open(os.path.join(workdir, "alf", "biases.json")) as fh:
        saved = json.load(fh)["alf_biases"]
    biases = alf_published.Biases(sum(nsubs), *(saved[k] for k in "bcxs"))

    runs = sorted(d for d in os.listdir(workdir)
                  if d == "run" or (d.startswith("run") and d[3:].isdigit()))
    lambda_runs = []
    for d in runs:
        path = os.path.join(workdir, d, "lambda.dat")
        if os.path.exists(path):
            lam = np.loadtxt(path)
            if lam.ndim == 2 and len(lam):
                lambda_runs.append(lam)
    if not lambda_runs:
        raise SystemExit(f"{workdir}: no run*/lambda.dat yet")

    # G_i = b_i - kT ln (P_i / N) per replica, on the absolute scale. The published estimator uses the raw
    # endpoint count, which is only valid when every run is the same length: a run of twice the length
    # lowers every G in it by kT ln 2, and the exponential average over replicas then weights the longer
    # run as though it were a lower free energy. Dividing by that run's own sample count removes it, which
    # makes replicas of different production lengths comparable. Within one replica the correction is the
    # same for every state, so it cancels in any ddG and nothing computed before this change moves.
    kT = alf_published.KB_KCAL * 300.0
    nblocks = sum(nsubs)
    starts = np.cumsum([0] + list(nsubs[:-1]))
    combos = [()]
    for s in range(len(nsubs)):
        combos = [c + (starts[s] + k,) for c in combos for k in range(nsubs[s])]
    G = np.zeros((len(lambda_runs), len(combos)))
    counts = np.zeros((len(lambda_runs), len(combos)), dtype=int)
    for j, combo in enumerate(combos):
        L = np.zeros(nblocks)
        L[list(combo)] = 1
        e_end = float(biases.energy(L)[0])
        for r, lam in enumerate(lambda_runs):
            P = int(np.sum(np.all(lam[:, list(combo)] > 0.99, axis=1)))
            counts[r, j] = P
            G[r, j] = -e_end - kT * np.log(P / len(lam)) if P > 0 else np.inf

    value = combine_replicas(G, kT)
    coverage = np.isfinite(G).sum(axis=0)    # replicas in which this state reached its endpoint
    error = np.full(len(combos), np.nan)
    if len(lambda_runs) > 1:                 # bootstrap over the replicas, on the absolute scale
        rng = np.random.default_rng(2401)
        nf = len(lambda_runs)
        boot = np.array([combine_replicas(G[rng.integers(0, nf, nf)], kT) for _ in range(1000)])
        # A resample that draws none of the replicas holding a state is UNDEFINED for that state, not
        # zero-width. np.std propagates the NaN, which silently throws away the error bar of every
        # state missing from even one replica -- with 3 replicas a given one is absent from (2/3)^3 =
        # 30% of resamples, so that is most of them. Take the spread over the defined resamples.
        # With coverage 1 every defined resample repeats the same single replica, so that spread is
        # exactly 0, which is not an error bar; those stay NaN and the table reports the coverage.
        ok = coverage >= 2
        if ok.any():
            with np.errstate(invalid="ignore"):
                error[ok] = np.nanstd(np.where(np.isfinite(boot[:, ok]), boot[:, ok], np.nan), axis=0)

    names = [n for site in prepared.mapping.sites for n in site.block_names]
    labels = ["+".join(names[i] for i in combo) for combo in combos]
    return labels, value, error, counts.sum(axis=0), len(lambda_runs), coverage


def analyze(net):
    free = os.path.join(HERE, f"out_{net}_free")
    cplx = os.path.join(HERE, f"out_{net}_complex")
    for wd in (free, cplx):
        if not os.path.isdir(wd):
            print(f"[{net}] {os.path.basename(wd)} is not there yet; skipping")
            return
    lab_f, g_f, e_f, n_f, reps_f, cov_f = leg_estimate(free)
    lab_c, g_c, e_c, n_c, reps_c, cov_c = leg_estimate(cplx)
    if lab_f != lab_c:
        raise SystemExit(f"[{net}] legs have different states:\n  free    {lab_f}\n  complex {lab_c}")

    with np.errstate(invalid="ignore"):
        ddg = g_c - g_f                                 # each leg on its own absolute scale
    err = np.sqrt(np.nan_to_num(e_c, nan=0.0) ** 2 + np.nan_to_num(e_f, nan=0.0) ** 2)
    err[np.isnan(e_c) | np.isnan(e_f)] = np.nan
    ref = lab_f.index(REFERENCE[net])
    if not np.isfinite(ddg[ref]):
        raise SystemExit(
            f"[{net}] the reference state {REFERENCE[net]} was not sampled in both legs "
            f"(endpoint counts: complex {int(n_c[ref])}, free {int(n_f[ref])}), so nothing can be "
            f"referenced to it. Run more production, or point REFERENCE at a state that was sampled.")
    ddg = ddg - ddg[ref]                                # now relative to the reference state

    unsampled = [lab_f[i] for i in range(len(lab_f)) if not np.isfinite(ddg[i])]

    measured = MEASURED[net]
    kd_ref = measured[REFERENCE[net]][1]
    print(f"\n=== network {net}: {len(lab_f)} states, {reps_f} free / {reps_c} complex replicas")
    print(f"ddG_bind in kcal/mol, relative to {measured[REFERENCE[net]][4]} "
          f"({REFERENCE[net]}); negative = binds MDM2 more tightly\n")
    print(f"{'state':14s} {'name':16s} {'MSLD':>14s} {'measured':>10s} {'Kd nM':>9s} "
          f"{'error':>7s} {'endpoint':>9s} {'reps':>6s}")
    rows = []
    for i, label in enumerate(lab_f):
        name = measured.get(label, ("", None, None, "", "(prediction)"))[4]
        if np.isfinite(ddg[i]):
            msld = f"{ddg[i]:+7.2f}" + (f" ± {err[i]:4.2f}" if np.isfinite(err[i]) else " ±  n/a")
        else:
            msld = "unsampled"           # the state never reached its endpoint in any replica
        exp_text, kd_text, delta_text = "", "", ""
        if label in measured:
            _, kd, sd, study, _ = measured[label]
            exp = RT * np.log(kd / kd_ref)
            exp_sd = RT * np.hypot(sd / kd, measured[REFERENCE[net]][2] / kd_ref)
            exp_text = f"{exp:+7.2f}"
            kd_text = f"{kd:9.4g}"
            if label != REFERENCE[net]:
                rows.append((label, name, ddg[i], err[i], exp, exp_sd, study))
                delta_text = f"{ddg[i] - exp:+7.2f}"
            if study != "L2021":
                name += "*"
        print(f"{label:14s} {name:16s} {msld:>14s} {exp_text:>10s} {kd_text:>9s} "
              f"{delta_text:>7s} {int(n_c[i]):5d}/{int(n_f[i]):d} "
              f"{int(cov_c[i])}/{int(cov_f[i])} of {reps_c}")
    print("  columns: MSLD = this calculation, measured = RT ln(Kd/Kd_ref), error = MSLD - measured,")
    print("           endpoint = endpoint samples in the complex/free leg, reps = replicas in which the")
    print("           state reached its endpoint.  * = cross-study row.")
    thin = [lab_f[i] for i in range(len(lab_f)) if np.isfinite(ddg[i]) and not np.isfinite(err[i])]
    if thin:
        print(f"\n  no error bar for {', '.join(thin)}: reached in only one replica of a leg, so a")
        print("  bootstrap over replicas has nothing to vary. The value is still the pooled estimate;")
        print("  it rests on one run, and another replica is what turns it into a number with a spread.")
    if unsampled:
        print(f"\n  {len(unsampled)} state(s) never reached an endpoint in one of the legs: "
              f"{', '.join(unsampled)}.")
        print("  These have no estimate. A zero count is not evidence of weak binding: it says the")
        print("  biases did not open that state, which is a statement about sampling, not affinity.")
        print("  More production, or more ALF cycles, is what fixes it.")

    if rows:
        d = np.array([r[2] - r[4] for r in rows])
        finite = np.isfinite(d)
        if finite.any():
            print(f"\n  {finite.sum()} measured states besides the reference: "
                  f"MAE {np.abs(d[finite]).mean():.2f}, RMSE {np.sqrt((d[finite] ** 2).mean()):.2f} kcal/mol, "
                  f"largest miss {np.abs(d[finite]).max():.2f}")
            m = np.array([r[2] for r in rows])[finite]
            x = np.array([r[4] for r in rows])[finite]
            if finite.sum() > 2:
                order_ok = np.corrcoef(np.argsort(np.argsort(m)), np.argsort(np.argsort(x)))[0, 1]
                print(f"  rank correlation with experiment (Spearman): {order_ok:+.2f}")
                print(f"  measured spread {x.max() - x.min():.2f} kcal/mol: a calculation worth "
                      f"believing has to be clearly better than that")

    if net in ADDITIVITY:
        singles, corner = ADDITIVITY[net]
        if corner in lab_f and all(s in lab_f for s in singles):
            total = sum(ddg[lab_f.index(s)] for s in singles)
            got = ddg[lab_f.index(corner)]
            print(f"\n  additivity: {' + '.join(singles)} = {total:+.2f}, {corner} = {got:+.2f}, "
                  f"so non-additivity {got - total:+.2f} kcal/mol")
            print("  The corner is TSFAEYWAQLMP, PMI-M3's 8-11 motif on the plain PMI backbone.")
            print("  Nothing measures it, so this is MSLD against itself. Experiment is anti-cooperative")
            print("  by +0.76 for PMI-M5 and +1.07 for PMI-M4 against their own singles, both on the")
            print("  L10A background, so a positive number of that order is what to expect.")

    out = os.path.join(HERE, f"report_{net}.json")
    with open(out, "w") as fh:
        json.dump({"states": lab_f, "ddG_bind_kcal": ddg.tolist(), "err_kcal": err.tolist(),
                   "reference": REFERENCE[net], "replicas": {"free": reps_f, "complex": reps_c},
                   "endpoint_counts": {"free": n_f.tolist(), "complex": n_c.tolist()},
                   "replica_coverage": {"free": cov_f.tolist(), "complex": cov_c.tolist()},
                   "measured": {k: v[1] for k, v in measured.items()}}, fh, indent=1)
    print(f"\n  wrote {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["val", "exp"]):
        if name not in MEASURED:
            raise SystemExit(f"unknown network {name!r}; expected val or exp")
        analyze(name)
