"""HybridSystemBuilder — hybrid ParmEd Structure -> plain OpenMM System.

The last structural step: turn the assembled (and, if requested, solvated) hybrid Structure
into a plain OpenMM `System` the engine can then lambda-scale. Two responsibilities:

  * createSystem with the right nonbonded settings. Because the Structure is ParmEd-native,
    createSystem is LOSSLESS -- it rebuilds LJ (incl. CHARMM NBFIX) from the atom types, so no
    recovery is needed here (that is only for the OpenMM->ParmEd direction, see ingest).
  * a safety net: refuse a topology whose bonded terms connect two blocks of the SAME site
    (illegal in MSLD). The mapper/assembler never create such a term; this guards externally
    supplied / pre-merged structures (reuses merge.assert_no_cross_variant_terms).

Same-site NONBONDED exclusions are deliberately not added here -- that stays in the engine, 
so the plain System remains a faithful, unexcluded parameter container and the
lambda=1==base invariant is testable against it.

Solvent helpers (`is_solvated` / `strip_solvent`) support the pre-solvated-input policy: 
assembly runs on a DRY solute, so a structure that already carries water/ions is stripped 
before grafting and re-solvated afterwards if asked.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import openmm as mm
from openmm import app, unit

from .merge import assert_no_cross_variant_terms
from .solvate import _is_water, _strip_residues


def _is_ion(res) -> bool:
    """A monatomic residue carrying a real charge (Na+, Cl-, K+, ...)."""
    return len(res.atoms) == 1 and abs(res.atoms[0].charge) > 0.5


def is_solvated(structure) -> bool:
    """True if the structure contains any water or ion residue."""
    return any(_is_water(r) or _is_ion(r) for r in structure.residues)


def strip_solvent(structure):
    """Return a copy with all water/ion residues removed (no-op if there are none)."""
    solvent = [r for r in structure.residues if _is_water(r) or _is_ion(r)]
    if not solvent:
        return structure
    out = copy.copy(structure)
    _strip_residues(out, [r for r in out.residues if _is_water(r) or _is_ion(r)])
    return out


@dataclass
class HybridSystemBuilder:
    """Build a plain OpenMM System from a hybrid Structure.

    nonbonded_method : an OpenMM app method, or None to auto-pick (PME if the structure has a
                       periodic box, else NoCutoff).
    nonbonded_cutoff : real-space cutoff (nm), used for the cutoff/PME methods.
    switch_distance  : LJ switching distance (nm), or None for no switching.
    constraints      : createSystem constraints (e.g. app.HBonds), or None.
    rigid_water      : keep water rigid (createSystem default).
    check_cross_variant : run the same-site cross-variant bonded-term guard first.
    """
    nonbonded_method: object = None
    nonbonded_cutoff: float = 1.0
    switch_distance: float | None = None
    constraints: object = None
    rigid_water: bool = True
    check_cross_variant: bool = True

    def build(self, structure, blueprint=None) -> mm.System:
        if self.check_cross_variant and blueprint is not None:
            sites = blueprint.sites if hasattr(blueprint, "sites") else blueprint
            assert_no_cross_variant_terms(structure, sites)

        method = self.nonbonded_method
        if method is None:
            method = app.PME if structure.box is not None else app.NoCutoff

        kw = dict(nonbondedMethod=method, constraints=self.constraints,
                  rigidWater=self.rigid_water)
        if method != app.NoCutoff:
            kw["nonbondedCutoff"] = self.nonbonded_cutoff * unit.nanometer
            if self.switch_distance is not None:
                kw["switchDistance"] = self.switch_distance * unit.nanometer
        return structure.createSystem(**kw)
