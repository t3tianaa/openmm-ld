"""Bonded-term utilities shared by the assembler: transplant + cross-variant guard.

Two independent helper groups, both used by the current prep pipeline:

  * TERM TRANSPLANT (used by assemble/structure.py): copy only a variant's block-touching
    bonded terms (bonds, angles, dihedrals, Urey-Bradley, impropers, CMAP, 1-4 "adjusts")
    onto the hybrid Structure, remapping atom indices and cloning each parameter type once.
  * CROSS-VARIANT GUARD (used by assemble/system.py, plus a public escape hatch): find /
    assert / delete bonded terms that illegally span two blocks of the SAME site
    (mutually-exclusive variants must never share a term).
"""
from __future__ import annotations

import copy

import parmed as pmd

from ..mapping.core import CoreMapping


# helpers
def _term_atoms(term):
    """Atoms referenced by a ParmEd bonded term (2 to 5 of them)."""
    out = []
    for attr in ("atom1", "atom2", "atom3", "atom4", "atom5"):
        a = getattr(term, attr, None)
        if a is not None:
            out.append(a)
    return out


def _copy_atom(src) -> "pmd.Atom":
    """A fresh ParmEd Atom copying src's identity, charge, mass, LJ and coordinates."""
    a = pmd.Atom(atomic_number=src.atomic_number, name=src.name, type=src.type,
                 charge=src.charge, mass=src.mass)
    # LJ: atom_type carries rmin/epsilon; also copy sigma/epsilon/rmin defensively
    a.atom_type = src.atom_type
    for attr in ("sigma", "epsilon", "rmin", "rmin_14", "epsilon_14"):
        val = getattr(src, attr, None)
        if val is not None:
            try:
                setattr(a, attr, val)
            except (AttributeError, TypeError):
                pass
    a.xx, a.xy, a.xz = src.xx, src.xy, src.xz
    return a


class _TypeCache:
    """Copies a variant's parameter-type objects into the base's type lists once each."""
    def __init__(self):
        self._seen: dict[int, object] = {}

    def get(self, t, base_list):
        if t is None:
            return None
        key = id(t)
        if key not in self._seen:
            nt = copy.copy(t)          # detach from the source structure
            base_list.append(nt)
            # the copy still points at the variant's type list, so its .idx would index THAT list;
            # Structure copy / `+` (e.g. solvate) rebuild terms from base.<types>[t.idx] and would
            # silently pick up an unrelated base type. Re-own every type in the base list.
            base_list.claim()
            self._seen[key] = nt
        return self._seen[key]


def _touches_block(atoms, block_src: set[int]) -> bool:
    return any(a.idx in block_src for a in atoms)


def _transplant_terms(base, src, amap, block_src, cache):
    """Copy src's block-touching bonded terms into base, remapping atom indices."""
    def A(term_atom):                          # source atom -> base atom object
        return base.atoms[amap[term_atom.idx]]

    # bonds
    for b in src.bonds:
        if _touches_block([b.atom1, b.atom2], block_src):
            t = cache.get(b.type, base.bond_types)
            base.bonds.append(pmd.Bond(A(b.atom1), A(b.atom2), type=t))

    # angles
    for ang in src.angles:
        ats = [ang.atom1, ang.atom2, ang.atom3]
        if _touches_block(ats, block_src):
            t = cache.get(ang.type, base.angle_types)
            base.angles.append(pmd.Angle(A(ats[0]), A(ats[1]), A(ats[2]), type=t))

    # proper + improper (periodic) dihedrals
    for d in src.dihedrals:
        ats = [d.atom1, d.atom2, d.atom3, d.atom4]
        if _touches_block(ats, block_src):
            t = cache.get(d.type, base.dihedral_types)
            base.dihedrals.append(pmd.Dihedral(
                A(ats[0]), A(ats[1]), A(ats[2]), A(ats[3]),
                improper=d.improper, ignore_end=d.ignore_end, type=t))

    # Urey-Bradley (CHARMM 1-3)
    for ub in getattr(src, "urey_bradleys", []):
        if _touches_block([ub.atom1, ub.atom2], block_src):
            t = cache.get(ub.type, base.urey_bradley_types)
            base.urey_bradleys.append(pmd.UreyBradley(A(ub.atom1), A(ub.atom2), type=t))

    # harmonic impropers (CHARMM)
    for imp in getattr(src, "impropers", []):
        ats = [imp.atom1, imp.atom2, imp.atom3, imp.atom4]
        if _touches_block(ats, block_src):
            t = cache.get(imp.type, base.improper_types)
            base.impropers.append(pmd.Improper(
                A(ats[0]), A(ats[1]), A(ats[2]), A(ats[3]), type=t))

    # CMAP (CHARMM backbone cross-term)
    for cm in getattr(src, "cmaps", []):
        ats = _term_atoms(cm)
        if _touches_block(ats, block_src):
            t = cache.get(cm.type, base.cmap_types)
            base.cmaps.append(pmd.Cmap(*[A(a) for a in ats], type=t))

    # 1-4 exceptions ("adjusts"): the scaled 1-4 pairs. These are not rederived by
    # createSystem when there are no dihedrals, so they must be transplanted, or the
    # env<->block 1-4 pair would wrongly become a full interaction.
    for adj in getattr(src, "adjusts", []):
        if _touches_block([adj.atom1, adj.atom2], block_src):
            t = cache.get(adj.type, base.adjust_types)
            base.adjusts.append(pmd.NonbondedException(A(adj.atom1), A(adj.atom2), type=t))


# ---------------------------------------------------------------------
# Cross-variant bonded-term check / delete
# ---------------------------------------------------------------------
# A bonded term (bond/angle/dihedral/UB/improper/CMAP/1-4) must NOT connect two blocks
# of the same site - those variants are mutually exclusive and never interact. Folowing
# helpers are a safety net for topologies built elsewhere (or supplied pre-merged). 
_TERM_LISTS = [("bond", "bonds"), ("angle", "angles"), ("dihedral", "dihedrals"),
               ("urey_bradley", "urey_bradleys"), ("improper", "impropers"),
               ("cmap", "cmaps"), ("adjust", "adjusts")]
_TERM_ATTR = dict(_TERM_LISTS)


def _atom_owner(mappings) -> dict:
    """atom index -> (site_index, block_name) for every block atom across mappings."""
    if isinstance(mappings, CoreMapping):
        mappings = [mappings]
    owner: dict[int, tuple[int, str]] = {}
    for site_i, m in enumerate(mappings):
        for name, atoms in m.blocks.items():
            for a in atoms:
                owner[a] = (site_i, name)
    return owner


def find_cross_variant_terms(structure, mappings) -> list:
    """Return [(category, atom_indices, term), ...] for terms that span two blocks of
    the same site. `mappings` is one CoreMapping (single site) or a list (one per site).
    """
    owner = _atom_owner(mappings)
    found = []
    for cat, attr in _TERM_LISTS:
        for term in getattr(structure, attr, []):
            idxs = [a.idx for a in _term_atoms(term)]
            per_site: dict[int, set] = {}
            for i in idxs:
                if i in owner:
                    site_i, name = owner[i]
                    per_site.setdefault(site_i, set()).add(name)
            if any(len(names) >= 2 for names in per_site.values()):
                found.append((cat, tuple(idxs), term))
    return found


def assert_no_cross_variant_terms(structure, mappings) -> None:
    """Raise if any bonded term connects two blocks of the same site."""
    found = find_cross_variant_terms(structure, mappings)
    if found:
        detail = "; ".join(f"{c}{list(idx)}" for c, idx, _ in found[:8])
        more = "" if len(found) <= 8 else f" (+{len(found) - 8} more)"
        raise ValueError(
            f"{len(found)} cross-variant bonded term(s) span two blocks of the same "
            f"site (illegal in MSLD): {detail}{more}. Remove them with "
            f"delete_cross_variant_terms() before building.")


def delete_cross_variant_terms(structure, mappings) -> int:
    """Delete every bonded term that spans two blocks of the same site. Returns the
    count removed. Safe to call when there are none (returns 0)."""
    found = find_cross_variant_terms(structure, mappings)
    for cat, _idx, term in found:
        term.delete()                              # detach from its atoms
        getattr(structure, _TERM_ATTR[cat]).remove(term)
    return len(found)
