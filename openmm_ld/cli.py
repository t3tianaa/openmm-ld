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
    "prepare": {"output": "prepared", "seed": 0, "renormalize": True, "solvate": True},
    "system": {"scale_bond": True, "scale_angle": True, "scale_torsion": True, "softcore_alpha": 0.5,
               "softcore_sigma_q": 0.3, "fnex": 5.5},
    "dynamics": {"temperature": 300.0, "timestep": 0.001, "atom_friction": 1.0, "theta_mass": 0.05,
                 "theta_friction": 1.0, "platform": "CPU", "threads": 4, "seed": 0},
    "minimize": {"max_iterations": 0},
    "alf": {"output": "alf", "iterations": 20, "steps_per_iter": 20000, "sample_every": 100, "rate": 1.0,
            "couple": True, "max_step": None, "tol": 0.05},
    "run": {"output": "run", "steps": 10000, "start": "alf", "metrics_every": 100, "trajectory_every": 1000},
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
    bundle = build_msld_system(environment=environment, sites=sites, forcefield=forcefield,
                               renormalize=cfg["renormalize"], solvate=cfg["solvate"])
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


def _flatten(dyn, model, config, workdir):
    import openmm as mm
    from .alf import estimate_ddg, flatten

    cfg = config["alf"]
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
    t = time.time()
    dyn.run(cfg["steps"], metrics=metrics, metrics_every=cfg["metrics_every"], mapping=prepared.mapping,
            trajectory=trajectory, trajectory_every=cfg["trajectory_every"], structure=prepared)
    elapsed = time.time() - t
    print(f"production: {cfg['steps']} steps ({elapsed:.0f} s, {elapsed / cfg['steps']:.3f} s/step); "
          f"T_theta {dyn.theta_temperature():.0f} K, T_atoms {dyn.atom_temperature():.0f} K", flush=True)
    if cfg["steps"] >= cfg["metrics_every"]:
        print("mean lambda:", {k: round(v, 3) for k, v in populations(metrics).items()}, flush=True)


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
