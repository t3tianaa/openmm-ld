"""The openmm-ld command: prepare a system, flatten it with ALF, run production. See docs/cli.md.

    openmm-ld prepare config.toml
    openmm-ld alf     config.toml     # minimize + ALF; writes the biases and the final state
    openmm-ld run     config.toml     # production, from the ALF state or the prepared system
    openmm-ld all     config.toml     # minimize, ALF and production in one process

Relative paths are against the config file; `--workdir` moves the outputs. `--set run.steps=100` overrides a value.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import tomllib

DEFAULTS = {
    "prepare": {"output": "prepared", "seed": 0, "renormalize": True, "solvate": True, "padding": 1.0,
                "ion_conc": 0.0, "water_ff": None, "cutoff": 1.0, "switch_distance": None, "constraints": None,
                "hydrogen_mass": None},
    "system": {"scale_bond": True, "scale_angle": True, "scale_torsion": True, "softcore_alpha": 0.5,
               "softcore_sigma_q": 0.3, "fnex": 5.5},
    "dynamics": {"temperature": 300.0, "timestep": 0.001, "atom_friction": 1.0, "theta_mass": 0.05,
                 "theta_friction": 1.0, "platform": "CPU", "threads": 4, "seed": 0},
    "minimize": {"max_iterations": 0},
    "alf": {"output": "alf", "method": "published", "iterations": 20, "steps_per_iter": 20000,
            "sample_every": 100, "rate": 1.0, "couple": True, "max_step": None, "tol": 0.05,
            "cycles": 50, "steps": 25000, "equil_steps": None, "equilibrate_steps": 0},
    "run": {"output": "run", "steps": 10000, "start": "alf", "metrics_every": 100, "trajectory_every": 1000,
            "equil_steps": 0, "lambda_every": 10, "chunks": 5, "cutoff": 0.99},
}


def load_config(path, overrides=()):
    with open(path, "rb") as fh:
        config = tomllib.load(fh)
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"{path}: unknown section(s) {sorted(unknown)}; expected {sorted(DEFAULTS)}")
    merged = {section: {**values, **config.get(section, {})} for section, values in DEFAULTS.items()}
    for item in overrides:
        key, _, text = item.partition("=")
        section, _, name = key.partition(".")
        if section not in merged or not name or not text:
            raise SystemExit(f"--set expects section.key=value, got {item!r}")
        try:
            merged[section][name] = tomllib.loads(f"v = {text}")["v"]
        except tomllib.TOMLDecodeError:
            merged[section][name] = text
    for section, values in merged.items():
        allowed = set(DEFAULTS[section]) | ({"forcefield", "environment", "site"} if section == "prepare" else set())
        extra = set(values) - allowed
        if extra:
            raise SystemExit(f"[{section}]: unknown key(s) {sorted(extra)}")
    return merged


def prepare(config, base, workdir):
    """Input files to a prepared system on disk."""
    from .prep import Site, build_msld_system, ingest

    cfg = config["prepare"]
    if not cfg.get("environment") or not cfg.get("site"):
        raise SystemExit("[prepare] needs at least one [[prepare.environment]] and one [[prepare.site]]")
    path = lambda name: os.path.join(base, name)
    forcefield = cfg.get("forcefield")
    random.seed(cfg["seed"])         # addHydrogens places hydrogens at random (docs/prep.md)
    t = time.time()
    parts, offsets, environment, count = [], [], None, 0
    for part in cfg["environment"]:
        structure = ingest(path(part["file"]), forcefield=part.get("forcefield", forcefield),
                           add_hydrogens=part.get("add_hydrogens", False))
        if "chain" in part:
            for residue in structure.residues:
                residue.chain = part["chain"]
        offsets.append(count)
        count += len(structure.residues)
        parts.append(structure)
    for structure in parts:
        environment = structure if environment is None else environment + structure
    sites = []
    for site in cfg["site"]:
        # `part` (from 0) says which environment file `residue` counts in; the default is the first
        residue = offsets[site.get("part", 0)] + site["residue"]
        variants = {name: path(file) for name, file in site["variants"].items()}
        sites.append(Site(residue=residue, variants=variants, mapper=site.get("mapper", "backbone")))
    water_ff = cfg["water_ff"]
    if water_ff is None:        # the water model must match the protein force field (prep/assemble/solvate.py)
        charmm = any("charmm" in str(name).lower() for name in (forcefield or []))
        water_ff = ["charmm36/water.xml"] if charmm else ["amber14/tip3p.xml"]
    solvate_kwargs = {"padding": cfg["padding"], "ion_conc": cfg["ion_conc"],
                      "water_ff": tuple([water_ff] if isinstance(water_ff, str) else water_ff)}
    system_kwargs = {"nonbonded_cutoff": cfg["cutoff"], "switch_distance": cfg["switch_distance"],
                     "constraints": cfg["constraints"], "hydrogen_mass": cfg["hydrogen_mass"]}
    bundle = build_msld_system(environment=environment, sites=sites, forcefield=forcefield,
                               renormalize=cfg["renormalize"], solvate=cfg["solvate"],
                               solvate_kwargs=solvate_kwargs, system_kwargs=system_kwargs)
    out = os.path.join(workdir, cfg["output"])
    bundle.save(out)
    print(f"prepared in {time.time() - t:.1f} s: {bundle.system.getNumParticles()} atoms, blocks "
          f"{[s.block_names for s in bundle.mapping.sites]}; wrote {out}", flush=True)


def _build(config, workdir, biases=None):
    """Load the prepared system and make the model, the scaled System and the dynamics."""
    from . import LambdaDynamics, load_prepared
    from .alf import add_flattening_biases

    sysc, dync = config["system"], config["dynamics"]
    prepared = load_prepared(os.path.join(workdir, config["prepare"]["output"]))
    options = {"theta_mass": dync["theta_mass"], "theta_friction": dync["theta_friction"]}
    names = [name for site in prepared.mapping.sites for name in site.block_names]
    model = prepared.to_model(fnex=sysc["fnex"], block_options={name: options for name in names})
    model.scale_bond, model.scale_angle = sysc["scale_bond"], sysc["scale_angle"]
    model.scale_torsion = sysc["scale_torsion"]
    model.softcore_alpha, model.softcore_sigma_q = sysc["softcore_alpha"], sysc["softcore_sigma_q"]
    published = config["alf"]["method"] == "published"
    if published:
        from . import alf_published
        alf_published.add_alf_biases(model)
        if biases:
            alf_published.apply_biases(model, alf_published.Biases(
                sum(alf_published.site_sizes(model)), *(biases["alf_biases"][k] for k in "bcxs")))
    else:
        add_flattening_biases(model, couple=config["alf"]["couple"])
        if biases:
            for b, value in biases["biases"].items():
                model.blocks[int(b)].lambda_bias = value
            for vb, k in zip(model.variable_biases, biases["couplings"]):
                vb.k = k
    info = model.scale_system(prepared.system)
    print(f"scaled: {prepared.system.getNumForces()} forces, {info['scaled_14_exceptions']} alchemical 1-4s", flush=True)
    properties = {"Threads": str(dync["threads"])} if dync["platform"] == "CPU" else {}
    dyn = LambdaDynamics(prepared.system, model, prepared.positions, temperature=dync["temperature"],
                         timestep=dync["timestep"], atom_friction=dync["atom_friction"], platform=dync["platform"],
                         properties=properties, box=prepared.box, seed=dync["seed"])
    return prepared, model, dyn


def _minimize(dyn, config):
    t = time.time()
    print(f"PE before minimize {dyn.potential_energy():.1f} kJ/mol", flush=True)
    dyn.minimize(max_iterations=config["minimize"]["max_iterations"])
    print(f"PE after minimize {dyn.potential_energy():.1f} kJ/mol ({time.time() - t:.0f} s); endpoint gaps "
          f"{({b: round(g, 1) for b, g in dyn.endpoint_gaps.items()})}", flush=True)


def _save_state(dyn, path):
    import openmm as mm
    state = dyn.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                 getIntegratorParameters=True)
    with open(path + ".tmp", "w") as fh:
        fh.write(mm.XmlSerializer.serialize(state))
    os.replace(path + ".tmp", path)


def _flatten_published(dyn, model, config, workdir):
    """Published ALF (openmm_ld.alf_published): cycles of `steps` sampled MD after `equil_steps`."""
    import openmm as mm
    from . import alf_published

    cfg = config["alf"]
    out = os.path.join(workdir, cfg["output"])
    os.makedirs(out, exist_ok=True)
    state_file = os.path.join(out, "state.xml")
    if os.path.exists(state_file):            # resume after an interruption
        with open(state_file) as fh:
            dyn.context.setState(mm.XmlSerializer.deserialize(fh.read()))
        dyn.invalidate_forces()
        print(f"resumed from {state_file}", flush=True)
    elif cfg["equilibrate_steps"]:
        t = time.time()
        dyn.step(int(cfg["equilibrate_steps"]))
        print(f"equilibrated {cfg['equilibrate_steps']} steps ({time.time() - t:.0f} s), T_atoms "
              f"{dyn.atom_temperature():.0f} K", flush=True)
    t = time.time()
    biases = alf_published.flatten(dyn, out, cycles=cfg["cycles"], steps=cfg["steps"],
                                   equil_steps=cfg["equil_steps"], sample_every=cfg["sample_every"],
                                   log=lambda m: print(m, flush=True),
                                   checkpoint=lambda d, i: _save_state(d, state_file))
    print(f"ALF (published): {cfg['cycles']} cycles ({time.time() - t:.0f} s); b (kcal/mol) "
          f"{[round(v, 2) for v in biases.b]}", flush=True)
    with open(os.path.join(out, "biases.json"), "w") as fh:
        json.dump({"method": "published", "alf_biases": biases.to_dict(), "units": "kcal/mol, ALF sign",
                   "blocks": [model.block_names[b] if hasattr(model, "block_names") else b
                              for b in range(1, model.n_blocks)]}, fh, indent=1)
    _save_state(dyn, state_file)


def _flatten(dyn, model, config, workdir):
    import openmm as mm
    from .alf import estimate_ddg, flatten

    cfg = config["alf"]
    if cfg["method"] == "published":
        return _flatten_published(dyn, model, config, workdir)
    t = time.time()
    report = flatten(dyn, couple=cfg["couple"], n_iters=cfg["iterations"], steps_per_iter=cfg["steps_per_iter"],
                     sample_every=cfg["sample_every"], rate=cfg["rate"], max_step=cfg["max_step"], tol=cfg["tol"])
    ddg = estimate_ddg(report, model)
    print(f"ALF: {report['iterations']} iterations ({time.time() - t:.0f} s), converged {report['converged']}, biases "
          f"{({b: round(v, 2) for b, v in report['biases'].items()})}", flush=True)
    print("DDG estimate (kJ/mol):", {b: round(v[0], 2) for b, v in ddg.items()}, flush=True)
    out = os.path.join(workdir, cfg["output"])
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "biases.json"), "w") as fh:
        json.dump({"biases": report["biases"], "couplings": report["couplings"], "converged": report["converged"],
                   "iterations": report["iterations"], "ddg": {b: list(v) for b, v in ddg.items()},
                   "history": report["history"], "bias_history": report["bias_history"],
                   "coupling_history": report["coupling_history"]}, fh, indent=1)
    state = dyn.context.getState(getPositions=True, getVelocities=True, getParameters=True, getIntegratorParameters=True)
    with open(os.path.join(out, "state.xml"), "w") as fh:
        fh.write(mm.XmlSerializer.serialize(state))


def _production(dyn, prepared, config, workdir):
    from .io import populations

    cfg = config["run"]
    out = os.path.join(workdir, cfg["output"])
    os.makedirs(out, exist_ok=True)
    metrics, trajectory = os.path.join(out, "metrics.csv"), os.path.join(out, "traj.dcd")
    if cfg["equil_steps"]:
        dyn.step(int(cfg["equil_steps"]))
    reporters = []
    if config["alf"]["method"] == "published":
        from .alf_published import _LambdaRecorder
        recorder = _LambdaRecorder(cfg["lambda_every"])
        reporters.append(recorder)
    t = time.time()
    dyn.run(cfg["steps"], metrics=metrics, metrics_every=cfg["metrics_every"], mapping=prepared.mapping,
            trajectory=trajectory, trajectory_every=cfg["trajectory_every"], structure=prepared,
            reporters=reporters)
    if reporters:
        _published_estimate(dyn, prepared, config, out, recorder)
    elapsed = time.time() - t
    print(f"production: {cfg['steps']} steps ({elapsed:.0f} s, {elapsed / cfg['steps']:.3f} s/step); "
          f"T_theta {dyn.theta_temperature():.0f} K, T_atoms {dyn.atom_temperature():.0f} K", flush=True)
    if cfg["steps"] >= cfg["metrics_every"]:
        print("mean lambda:", {k: round(v, 3) for k, v in populations(metrics).items()}, flush=True)


def _published_estimate(dyn, prepared, config, out, recorder):
    """ALF's production estimator on this run: G = b - kT ln P(lambda > cutoff), bootstrap over chunks."""
    import numpy as np
    from . import alf_published

    cfg = config["run"]
    lam = np.array(recorder.rows)
    np.savetxt(os.path.join(out, "lambda.dat"), lam, fmt="%10.6f")
    model = dyn.model
    nsubs = alf_published.site_sizes(model)
    with open(os.path.join(workdir_of(out), config["alf"]["output"], "biases.json")) as fh:
        saved = json.load(fh)["alf_biases"]
    biases = alf_published.Biases(sum(nsubs), *(saved[k] for k in "bcxs"))
    chunks = [c for c in np.array_split(lam, max(1, int(cfg["chunks"]))) if len(c)]
    result = alf_published.endpoint_free_energies(chunks, biases, nsubs, dyn.temperature, cutoff=cfg["cutoff"])
    names = [n for site in prepared.mapping.sites for n in site.block_names]
    result["labels"] = ["+".join(names[i] for i in combo) for combo in result["combos"]]
    result["dG_kJ"] = [v * alf_published.KCAL for v in result["dG_kcal"]]
    result["err_kJ"] = [v * alf_published.KCAL for v in result["err_kcal"]]
    result["frac_endpoint"] = (lam > cfg["cutoff"]).mean(axis=0).tolist()
    with open(os.path.join(out, "dG.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print("dG vs first block (kJ/mol):", {l: f"{g:.2f} +/- {e:.2f}" for l, g, e in
                                          zip(result["labels"], result["dG_kJ"], result["err_kJ"])}, flush=True)


def workdir_of(out):
    return os.path.dirname(os.path.abspath(out))


def ddg(complex_json, free_json, out=None):
    with open(complex_json) as fh:
        cplx = json.load(fh)
    with open(free_json) as fh:
        free = json.load(fh)
    if cplx["labels"] != free["labels"]:
        raise SystemExit(f"legs have different states: {cplx['labels']} vs {free['labels']}")
    rows = []
    for l, gc, ec, gf, ef in zip(cplx["labels"], cplx["dG_kJ"], cplx["err_kJ"], free["dG_kJ"], free["err_kJ"]):
        d, e = gc - gf, (ec ** 2 + ef ** 2) ** 0.5
        rows.append({"state": l, "ddG_kJ": d, "err_kJ": e, "ddG_kcal": d / 4.184, "err_kcal": e / 4.184,
                     "complex_kJ": gc, "free_kJ": gf})
        print(f"{l:12s} ddG_bind = {d:8.2f} +/- {e:5.2f} kJ/mol  ({d / 4.184:6.2f} +/- {e / 4.184:4.2f} kcal/mol)")
    if out:
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=1)
    return rows


def alf(config, base, workdir):
    prepared, model, dyn = _build(config, workdir)
    _minimize(dyn, config)
    _flatten(dyn, model, config, workdir)


def run(config, base, workdir):
    import openmm as mm

    start = config["run"]["start"]
    if start == "alf":
        out = os.path.join(workdir, config["alf"]["output"])
        with open(os.path.join(out, "biases.json")) as fh:
            biases = json.load(fh)
        prepared, model, dyn = _build(config, workdir, biases)
        with open(os.path.join(out, "state.xml")) as fh:
            dyn.context.setState(mm.XmlSerializer.deserialize(fh.read()))
        dyn.invalidate_forces()
    elif start == "prepared":
        prepared, model, dyn = _build(config, workdir)
        _minimize(dyn, config)
    else:
        raise SystemExit(f"[run] start must be 'alf' or 'prepared', got {start!r}")
    _production(dyn, prepared, config, workdir)


def run_all(config, base, workdir):
    """Minimize, ALF and production in one process, one random stream."""
    prepared, model, dyn = _build(config, workdir)
    _minimize(dyn, config)
    _flatten(dyn, model, config, workdir)
    _production(dyn, prepared, config, workdir)


COMMANDS = {"prepare": prepare, "alf": alf, "run": run, "all": run_all}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "ddg":
        if len(argv) < 3:
            raise SystemExit("usage: openmm-ld ddg <complex run/dG.json> <free run/dG.json> [out.json]")
        ddg(argv[1], argv[2], argv[3] if len(argv) > 3 else None)
        return 0
    parser = argparse.ArgumentParser(prog="openmm-ld", description="Multisite lambda dynamics on OpenMM.")
    parser.add_argument("command", choices=sorted(COMMANDS), help="what to do")
    parser.add_argument("config", help="TOML config file")
    parser.add_argument("--workdir", help="where outputs go (default: next to the config file)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="override a config value")
    args = parser.parse_args(argv)
    base = os.path.dirname(os.path.abspath(args.config))
    workdir = os.path.abspath(args.workdir) if args.workdir else base
    os.makedirs(workdir, exist_ok=True)
    COMMANDS[args.command](load_config(args.config, args.set), base, workdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
