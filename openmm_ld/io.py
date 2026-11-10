"""Trajectory / results output for lambda-dynamics runs (ported from OpenMM-MSLD msld/io.py).

`LambdaDynamics` samples atoms + theta; this module writes the run's *observables* — per-block
lambda + potential energy (+ theta temperature) — to disk as the run proceeds. It is the minimal
first piece of the deferred io layer: the lambda/energy trajectory + a populations summary. Free
energy (DeltaG) estimation is NOT here yet (needs ALF flattening + analysis; see vault FEAT-040 / FEAT-012).

Each writer carries its own step `every` interval and manages its own file, so a run just names
the outputs and the intervals -- no `with` needed (`LambdaDynamics.run` opens nothing and closes
every reporter for you):

    dyn.run(10_000,
            metrics="out/metrics.csv",    metrics_every=100,   mapping=bundle.mapping,
            trajectory="out/traj.dcd",    trajectory_every=500, structure=bundle.structure)
    print(populations("out/metrics.csv"))     # mean lambda per block == raw variant populations

The reporters (`LambdaTrajectoryWriter`, `TrajectoryWriter`) are also usable directly via
`dyn.run(nsteps, reporters=[...])` if you want to build them yourself; each has `.every`,
`.report(dyn, step)`, and `.close()`.
"""
from __future__ import annotations

import os

from .forces import lambda_name


def block_labels(model, mapping=None) -> list[str]:
    """Human label for each non-environment block, in model (== column) order.

    Uses a HybridBlueprint `mapping`'s per-site variant names when given (site-prefixed as
    `s{k}:{name}` if there is more than one site); otherwise falls back to `lambda{b}`. Falls
    back too if the mapping's block count does not match the model's.
    """
    n_blocks = model.n_blocks
    if mapping is not None and getattr(mapping, "sites", None):
        labels: list[str] = []
        multi = len(mapping.sites) > 1
        for k, site in enumerate(mapping.sites, start=1):
            for name in site.block_names:
                labels.append(f"s{k}:{name}" if multi else name)
        if len(labels) == n_blocks - 1:            # one label per non-env block
            return labels
    return [lambda_name(b) for b in range(1, n_blocks)]


class LambdaTrajectoryWriter:
    """Append per-block lambda + energy to a CSV every `every` steps (a reporter).

    Columns: `step, time_ps, potential_kJ, theta_temp_K,` then one column per non-environment
    block (labeled via `block_labels`). `report(dyn, step)` writes one row and flushes (so a long
    run's results survive a crash). `every` is the step interval `LambdaDynamics.run` reports at.
    The file is opened here and closed by `run` (or `close()`); a context manager is also allowed.
    """

    def __init__(self, path, model, every: int = 100, mapping=None):
        self.path = path
        self.model = model
        self.every = int(every)
        self.blocks = list(range(1, model.n_blocks))      # non-env block indices
        self.labels = block_labels(model, mapping)
        self._f = open(path, "w")
        self._f.write("step,time_ps,potential_kJ,theta_temp_K," + ",".join(self.labels) + "\n")
        self._f.flush()

    def report(self, dyn, step: int) -> None:
        """Write one row for the current state of `dyn` (a LambdaDynamics) at `step`."""
        lam = dyn.get_lambdas()
        row = [step, step * dyn.dt, dyn.potential_energy(), dyn.theta_temperature()]
        row += [lam[b] for b in self.blocks]
        self._f.write(",".join(f"{v:.8g}" for v in row) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> "LambdaTrajectoryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class TrajectoryWriter:
    """Append atom coordinates to a DCD trajectory every `every` steps (a reporter).

    `structure` is the hybrid ParmEd Structure (used only for its OpenMM `topology` + initial
    positions). On the first frame it writes a companion topology PDB (`<stem>.pdb`) so the DCD is
    loadable in a viewer, then appends `dyn.x` each interval. The DCD is opened lazily on the first
    report (so the per-frame time = timestep * every is known) and closed by `LambdaDynamics.run`.

    The companion PDB keeps the DCD's atom order (so it indexes the frames), which means it is the
    RAW topology, not the merged-HYB rendering of `hybrid_output.write_hybrid_pdb` (that one reorders
    atoms). The raw topology appends each competing variant's side chain as its OWN residue named
    after the standard amino acid but holding only the side-chain fragment (no backbone). Left as-is,
    `PDBFile.writeFile` treats those as standard residues and writes NO CONECT, so a viewer misapplies
    the amino-acid template and cannot bond the fragment to the shared CB -> the side chains render as
    free-floating atoms. Passing `mapping` fixes this: the block-only residues are renamed to
    `hyb_resname` (a non-standard name) just for the PDB, which makes `writeFile` emit CONECT for
    their bonds and stops the template misapplication. Atom order is unchanged.
    """

    def __init__(self, path, structure, every: int = 1000, mapping=None, hyb_resname="HYB"):
        self.path = path
        self.structure = structure
        self.every = int(every)
        self.mapping = mapping
        self.hyb_resname = hyb_resname
        self._file = None
        self._dcd = None

    def _block_atom_indices(self) -> set:
        """Combined-index set of every non-environment (block) atom, from the blueprint.

        A site is skipped if it does not expose `all_block_atoms()` (e.g. a metrics-only mapping
        stub) or `blocks`, so a partial mapping just means no renaming rather than an error.
        """
        idx: set = set()
        for site in getattr(self.mapping, "sites", []) or []:
            if hasattr(site, "all_block_atoms"):
                idx.update(site.all_block_atoms())
            elif hasattr(site, "blocks"):
                for atoms in site.blocks.values():
                    idx.update(atoms)
        return idx

    def _open(self, dyn) -> None:
        from openmm import unit
        from openmm.app import DCDFile, PDBFile
        top = self.structure.topology

        # rename block-only residues to a non-standard name so writeFile emits CONECT for the
        # alchemical side chains (see the class docstring); restore the names afterwards so the
        # caller's structure is untouched.
        renamed = []
        if self.mapping is not None:
            block = self._block_atom_indices()
            if block:
                for res in top.residues():
                    ids = [a.index for a in res.atoms()]
                    if ids and all(i in block for i in ids):
                        renamed.append((res, res.name))
                        res.name = self.hyb_resname
        try:
            with open(os.path.splitext(self.path)[0] + ".pdb", "w") as fh:      # topology, once
                PDBFile.writeFile(top, self.structure.positions, fh)
        finally:
            for res, name in renamed:
                res.name = name

        self._file = open(self.path, "wb")
        self._dcd = DCDFile(self._file, top, dyn.dt * self.every * unit.picosecond)

    def report(self, dyn, step: int) -> None:
        from openmm import unit
        if self._dcd is None:
            self._open(dyn)
        self._dcd.writeModel(unit.Quantity(dyn.x, unit.nanometer))         # dyn.x is (N,3) in nm

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self._file.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------
# read-back helpers
# ---------------------------------------------------------------------
def read_trajectory(path):
    """Read a written CSV back. Returns (labels, columns) where `columns` maps each header name
    to a numpy array and `labels` is just the per-block lambda columns. numpy required."""
    import numpy as np
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    columns = {name: data[:, i] for i, name in enumerate(header)}
    labels = header[4:]                                # after step,time_ps,potential_kJ,theta_temp_K
    return labels, columns


def populations(path) -> dict:
    """Mean lambda per block label over a written trajectory (the raw variant 'populations')."""
    import numpy as np
    labels, columns = read_trajectory(path)
    return {label: float(np.mean(columns[label])) for label in labels}
