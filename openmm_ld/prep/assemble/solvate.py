"""Solvation + ions + box.

Neither tleap (Amber) nor OpenMM's Modeller.addSolvent works directly on our combined
topology: tleap may be absent, and addSolvent needs a ForceField template for every
solute residue, which our alchemical / non-canonical blocks do not have. So we solvate by
overlay:

  1. build a pure pre-equilibrated water box (empty solute -> no template needed),
  2. overlay it on the solute and delete waters that clash with any solute atom,
  3. add counter-ions to neutralize the solute's net charge (convert far waters to ions).

What is force-field-agnostic here is only the water placement: steps 1-2 need no template
for the solute -- water goes in by geometry (a clash distance), never consulting the
solute's force field. That is the single thing addSolvent/tleap could not do for a
non-templated solute. The parameters are not agnostic: the solute keeps its own ParmEd
parameters, and water + ions get parameters from a chosen water force field (`water_ff`).

WATER/ION FORCE FIELD MUST MATCH THE SOLUTE'S FORCE FIELD. This is a correctness
requirement, not a convenience:
  * ions differ by force field (Amber Joung-Cheatham vs CHARMM); a mismatch gives wrong
    ion-solute interactions,
  * the water model must be the one the solute FF was validated with.
      Amber ff14SB  -> amber14/tip3p.xml
      Amber ff19SB  -> amber14/opc.xml     (ff19SB was fit with OPC, NOT TIP3P)
The default (amber14/tip3p) suits an Amber-ff14SB solute; pass `water_ff` otherwise.

CHARMM water: supported and exact (ions included). CHARMM puts LJ in a tabulated
CustomNonbondedForce (acoef/bcoef by atom type) that ParmEd's load_topology drops, so we
recover it: the diagonal gives each atom's self LJ (sigma=(A/B)^(1/6), eps=B^2/4A), and the
off-diagonal, wherever it deviates from Lorentz-Berthelot, is put back as an NBFIX on the
ParmEd atom types -- so the ion-water NBFIX is preserved, not dropped. ParmEd rebuilds the
same tabulated force on createSystem. (Pass water_ff=("charmm36/water.xml",) for CHARMM.)

Coordinates: ParmEd is in angstrom, OpenMM in nm; `padding` and `clash` are given in nm.
Conversions happen only at the library boundary (Quantity where possible, explicit *10
for bare numpy arrays). Clash detection uses a scipy KD-tree (cKDTree) -- fine for Tier-1
sizes; a periodic cell list would be needed for very large systems.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import app, unit
from scipy.spatial import cKDTree                       # fast neighbor search for clashes

from ..ingest.recover import recover_charmm_lj


def _clash_mask(solute_xyz, solvent_xyz, clash_ang):
    """Boolean per solvent atom: True if within `clash_ang` of any solute atom.

    KD-tree neighbor search (O(M log N), scipy cKDTree). The solute sits centered/interior in
    the box, so non-periodic distances are correct here (its periodic images are near the box
    corners, far from the solute cavity). OpenMM's own addSolvent uses a periodic cell-list.
    """
    dist, _ = cKDTree(solute_xyz).query(solvent_xyz, distance_upper_bound=clash_ang)
    return dist < clash_ang                           # dist is +inf when nothing is near


def _load_topology(topology, system, xyz):
    """ParmEd load_topology, silencing its expected 'Unsupported CustomNonbondedForce'
    warning -- for CHARMM water we drop that force here and recover its LJ afterwards."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unsupported Force type")
        return pmd.openmm.load_topology(topology, system=system, xyz=xyz)

_NM = unit.nanometer
_MOLAR = unit.molar


@dataclass
class SolvateResult:
    structure: "pmd.Structure"     # solute + water + ions, with a periodic box
    n_waters: int
    n_pos_ions: int
    n_neg_ions: int
    box_nm: float
    net_charge: float              # total system charge (should be ~0)


def _is_water(res) -> bool:
    return len(res.atoms) == 3 and any(a.atomic_number == 8 for a in res.atoms)


def _water_oxygen_pos(res) -> np.ndarray:
    o = next(a for a in res.atoms if a.atomic_number == 8)
    return np.array([o.xx, o.xy, o.xz])


def _ion_templates(ff, water_model, pos_ion, neg_ion):
    """Return (positive, negative) reference ParmEd ion atoms to clone from."""
    mod = app.Modeller(app.Topology(), np.zeros((0, 3)) * _NM)
    mod.addSolvent(ff, model=water_model, boxSize=mm.Vec3(2.2, 2.2, 2.2) * _NM,
                   ionicStrength=0.5 * _MOLAR, positiveIon=pos_ion, negativeIon=neg_ion)
    sysm = ff.createSystem(mod.topology, nonbondedMethod=app.PME, rigidWater=False)
    st = _load_topology(mod.topology, sysm,
                        np.array(mod.positions.value_in_unit(_NM)) * 10.0)
    recover_charmm_lj(st, sysm)                  # CHARMM: restore LJ ParmEd dropped
    pos = neg = None
    for a in st.atoms:
        if len(a.residue.atoms) == 1:            # single-atom residue = ion
            if a.charge > 0 and pos is None:
                pos = a
            elif a.charge < 0 and neg is None:
                neg = a
    if pos is None or neg is None:
        raise RuntimeError("could not extract ion templates from the water force field")
    return pos, neg


def _clone_ion(template, pos_ang) -> "pmd.Atom":
    a = pmd.Atom(atomic_number=template.atomic_number, name=template.name,
                 type=template.type, charge=template.charge, mass=template.mass)
    a.atom_type = template.atom_type
    a.xx, a.xy, a.xz = float(pos_ang[0]), float(pos_ang[1]), float(pos_ang[2])
    return a


def _strip_residues(struct, residues):
    """Strip the given residues (by object) from a ParmEd Structure."""
    mask = np.zeros(len(struct.atoms), dtype=bool)
    for r in residues:
        for a in r.atoms:
            mask[a.idx] = True
    if mask.any():
        struct.strip(mask)


def solvate(solute, *, padding: float = 1.0, water_model: str = "tip3p",
            water_ff=("amber14/tip3p.xml",), ion_conc: float = 0.0,
            clash: float = 0.20, positive_ion: str = "Na+",
            negative_ion: str = "Cl-", neutralize: bool = True,
            net_charge: float | None = None) -> SolvateResult:
    """Solvate a dry solute Structure in a cubic water box with neutralizing ions.

    padding   : min distance (nm) from the solute to each box wall.
    water_ff  : ffxml(s) giving the water + ion parameters. MUST MATCH the solute's force
                field (see module docstring): amber14/tip3p.xml (ff14SB),
                amber14/opc.xml (ff19SB), or charmm36/water.xml (CHARMM36). Passing the
                wrong one silently produces a physically inconsistent system.
    water_model: the water template name inside `water_ff` (e.g. 'tip3p', 'opc').
    ion_conc  : background salt concentration (molar); 0 = only neutralizing ions.
    clash     : delete a water if any of its atoms is within this distance (nm) of a
                solute atom.
    net_charge: the solute's physical net charge, used to choose the neutralizing ions. None
                sums every atom, which is right only for a plain solute: a hybrid carries every
                variant's block at once, so the assembler passes the charge of one endpoint.
    Returns a SolvateResult whose .structure has a periodic box set.
    """
    if solute.coordinates is None:
        raise ValueError("solute has no coordinates")
    solute = copy.copy(solute)                    # do not mutate the caller's structure

    coords = np.array(solute.coordinates)         # angstrom (N,3)
    center = (coords.min(0) + coords.max(0)) / 2
    extent = float((coords.max(0) - coords.min(0)).max())
    box_ang = extent + 2 * padding * 10.0         # cubic side (angstrom)
    solute.coordinates = coords - center          # center solute at the origin
    box_nm = box_ang / 10.0

    ff = app.ForceField(*water_ff)

    # 1. pure water box (+ background salt), centered at the origin like the solute
    mod = app.Modeller(app.Topology(), np.zeros((0, 3)) * _NM)
    mod.addSolvent(ff, model=water_model, boxSize=mm.Vec3(box_nm, box_nm, box_nm) * _NM,
                   ionicStrength=ion_conc * _MOLAR,
                   positiveIon=positive_ion, negativeIon=negative_ion)
    solv_sys = ff.createSystem(mod.topology, nonbondedMethod=app.PME, rigidWater=False)
    solv = _load_topology(mod.topology, solv_sys,
                          np.array(mod.positions.value_in_unit(_NM)) * 10.0)
    recover_charmm_lj(solv, solv_sys)   # CHARMM: restore LJ ParmEd dropped

    # 2. delete solvent residues clashing with the solute (KD-tree neighbor search)
    clash_atom = _clash_mask(np.array(solute.coordinates), np.array(solv.coordinates),
                             clash * 10.0)
    clashing = [r for r in solv.residues if any(clash_atom[a.idx] for a in r.atoms)]
    _strip_residues(solv, clashing)

    # 3. neutralize: convert the waters farthest from the solute into counter-ions
    n_pos_added = n_neg_added = 0
    if neutralize:
        q_solute = (sum(a.charge for a in solute.atoms) if net_charge is None
                    else float(net_charge))
        q_total = round(q_solute + sum(a.charge for a in solv.atoms))
        if q_total != 0:
            pos_tpl, neg_tpl = _ion_templates(ff, water_model, positive_ion, negative_ion)
            template = neg_tpl if q_total > 0 else pos_tpl
            k = abs(q_total)
            waters = [r for r in solv.residues if _is_water(r)]
            su = np.array(solute.coordinates)
            far = sorted(waters,
                         key=lambda r: -((_water_oxygen_pos(r)[None, :] - su) ** 2).sum(-1).min())
            chosen = far[:k]
            ion_positions = [_water_oxygen_pos(r) for r in chosen]
            _strip_residues(solv, chosen)
            ions = pmd.Structure()
            for p in ion_positions:
                ions.add_atom(_clone_ion(template, p), template.residue.name,
                              len(ions.residues) + 1)
            solv = solv + ions
            if q_total > 0:
                n_neg_added = k
            else:
                n_pos_added = k

    combined = solute + solv
    combined.box = [box_ang, box_ang, box_ang, 90.0, 90.0, 90.0]

    n_waters = sum(1 for r in combined.residues if _is_water(r))
    n_pos = sum(1 for r in combined.residues if len(r.atoms) == 1 and r.atoms[0].charge > 0)
    n_neg = sum(1 for r in combined.residues if len(r.atoms) == 1 and r.atoms[0].charge < 0)
    net = sum(a.charge for a in combined.atoms)
    return SolvateResult(structure=combined, n_waters=n_waters, n_pos_ions=n_pos,
                         n_neg_ions=n_neg, box_nm=box_nm, net_charge=net)
