"""Inspection artifacts for a hybrid system: a merged single-residue PDB (+ CONECT) and a
self-contained OpenMM-style force-field XML for each hybrid residue.

The assembler keeps a variant's block atoms in separate ParmEd residues -- the engine only needs
the atom->block INDEX map, not residue grouping. But that makes the raw topology impossible to read
(the variant side chains stack as chain-less residues after the chain, with no CONECT). This module
renders the CORRECT picture instead:

  * write_hybrid_pdb   -- every atom at a mutated site (the shared CORE + ALL variants' side chains)
                          as ONE residue (default name HYB), contiguous, on the site's chain, with
                          explicit CONECT (it is a novel residue, so bonds must be spelled out).
  * write_hybrid_ff_xml -- a self-contained forcefield XML for that residue: <AtomTypes>, a
                          <Residue> (per-atom name/type/charge + bonds), and the bonded + nonbonded
                          parameter sections. Parameters are read from a plain OpenMM System (built
                          from the hybrid Structure), so every number is EXACT and in OpenMM units
                          -- no hand unit-conversion.

Both take the hybrid `structure` (ParmEd) + its `blueprint` (HybridBlueprint; one global-index
CoreMapping per site). Used by HybridSystemBundle.save.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict

import openmm as mm
from openmm import app, unit


# ---------------------------------------------------------------------
# shared: figure out each site's hybrid residue (which atoms, ordered, where)
# ---------------------------------------------------------------------
def _element_symbol(z) -> str:
    try:
        return app.Element.getByAtomicNumber(int(z)).symbol
    except Exception:
        return ""


def _hyb_sites(structure, blueprint, hyb_resname="HYB"):
    """Per site: {resname, number, chain, core_res_idx, atoms (ordered global idx), atomset, block}.

    A site's hybrid residue = its shared env core + every block atom (all variants). The env core
    lives in the original site residue; the other variants' block atoms are appended elsewhere. We
    order them core-residue-first, then the appended block atoms (structure order), so each variant
    stays contiguous.
    """
    sites = []
    n = len(blueprint.sites)
    for k, m in enumerate(blueprint.sites):
        env_atoms = set(m.env_atoms)
        block_atoms = set(m.all_block_atoms())
        core_res = structure.atoms[next(iter(env_atoms))].residue
        core_ids = [a.idx for a in core_res.atoms]                 # env core + WT block, in order
        rest = sorted(i for i in (env_atoms | block_atoms) if i not in set(core_ids))
        name = hyb_resname if n == 1 else f"{hyb_resname[:2]}{k + 1}"
        sites.append(dict(k=k, resname=name[:4], number=core_res.number, chain=core_res.chain,
                          core_res_idx=core_res.idx, atoms=core_ids + rest,
                          atomset=env_atoms | block_atoms, block=block_atoms))
    return sites


# ---------------------------------------------------------------------
# PDB
# ---------------------------------------------------------------------
def _pdb_name(name) -> str:
    """PDB atom-name field (cols 13-16): element-justified 4 chars."""
    name = str(name)
    return name[:4] if len(name) >= 4 else (" " + name).ljust(4)


_WATER_RESNAMES = {"HOH", "WAT", "TIP", "TIP3", "TIP4", "TIP5", "T3P", "TP3", "SPC", "SOL"}


def _is_solvent_res(res) -> bool:
    """A water or monatomic-ion residue (used to suppress TER between solvent molecules)."""
    return res.name in _WATER_RESNAMES or len(res.atoms) == 1


def _unique4(name, used) -> str:
    """A <=4-char PDB atom name derived from `name` that is not already in `used`.

    The merged residue disambiguates overlaid variants by APPENDING a digit (HD11 -> HD111), which
    overflows the 4-char PDB atom-name field and truncates back to a collision (HD111 -> HD11). Here
    we re-disambiguate WITHIN 4 characters: truncate, and if that collides, vary the last character.
    Names are just labels (CONECT uses serial numbers), so any unique 4-char spelling is fine.
    """
    cand = str(name)[:4]
    if cand not in used:
        return cand
    for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        alt = (str(name)[:3] + c)[:4]
        if alt not in used:
            return alt
    for k in range(1000):                                          # last resort: 2-char stem + counter
        alt = ("%s%d" % (str(name)[:2], k))[:4]
        if alt not in used:
            return alt
    raise ValueError("cannot make a unique 4-char PDB name for %r" % name)


def write_hybrid_pdb(structure, blueprint, path, hyb_resname="HYB") -> str:
    """Write a PDB where each site's atoms are merged into ONE residue, with CONECT. Returns path.

    Faithful to the assembled structure: writes a CRYST1 box for a periodic (solvated) system, a TER
    at every real chain break (consecutive residues NOT joined by a bond -- so a docked peptide is
    not shown covalently fused to the protein), and atom names made unique within the 4-char PDB
    field (so overlaid variant hydrogens do not collide after truncation).
    """
    sites = _hyb_sites(structure, blueprint, hyb_resname)
    by_core = {s["core_res_idx"]: s for s in sites}
    block_atom_site = {i: s for s in sites for i in s["block"]}
    hyb_global = set().union(*[s["atomset"] for s in sites]) if sites else set()

    # output order: (global idx, out resname, out resnum, out chain, is_hyb, source_res_idx)
    out = []
    for r in structure.residues:
        if r.idx in by_core:                                       # the site residue -> emit HYB
            s = by_core[r.idx]
            for i in s["atoms"]:
                out.append((i, s["resname"], s["number"], s["chain"], True, r.idx))
            continue
        ratoms = [a.idx for a in r.atoms]
        if ratoms and all(i in block_atom_site for i in ratoms):   # appended block residue -> skip
            continue
        for a in r.atoms:
            out.append((a.idx, r.name, r.number, r.chain, False, r.idx))

    serial = {gid: n + 1 for n, (gid, *_) in enumerate(out)}

    # residues joined by a bond (for TER at real chain breaks, not just chain-label changes)
    res_bonded = set()
    for b in structure.bonds:
        ri, rj = b.atom1.residue.idx, b.atom2.residue.idx
        if ri != rj:
            res_bonded.add(frozenset((ri, rj)))

    lines = []
    if structure.box is not None:                                  # periodic -> write the box
        a, bb, c, al, be, ga = (float(v) for v in structure.box[:6])
        lines.append("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f %-11s%4d\n"
                     % (a, bb, c, al, be, ga, "P 1", 1))

    prev_chain = prev_res = None
    used_names: set = set()
    for gid, resname, resnum, chain, is_hyb, residx in out:
        a = structure.atoms[gid]
        ch = (chain or "A")[:1]
        if prev_res is not None and residx != prev_res:            # residue boundary
            used_names = set()
            both_solvent = (_is_solvent_res(structure.residues[prev_res])
                            and _is_solvent_res(structure.residues[residx]))
            chain_break = frozenset((prev_res, residx)) not in res_bonded and not both_solvent
            if ch != prev_chain or chain_break:
                lines.append("TER\n")
        name = _unique4(a.name, used_names)
        used_names.add(name)
        lines.append("%-6s%5d %s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n" % (
            "HETATM" if is_hyb else "ATOM  ", serial[gid] % 100000, _pdb_name(name),
            str(resname)[:3], ch, resnum % 10000, a.xx, a.xy, a.xz, 1.0, 0.0,
            _element_symbol(a.atomic_number).rjust(2)))
        prev_chain, prev_res = ch, residx
    lines.append("TER\n")

    # CONECT for every bond touching a hybrid-residue atom (a novel residue: bonds must be explicit)
    adj = defaultdict(set)
    for b in structure.bonds:
        i, j = b.atom1.idx, b.atom2.idx
        if i in hyb_global or j in hyb_global:
            adj[i].add(j); adj[j].add(i)
    for i in sorted(adj):
        partners = sorted(serial[j] for j in adj[i] if j in serial)
        for c in range(0, len(partners), 4):
            lines.append("CONECT%5d%s\n" % (serial[i], "".join("%5d" % p for p in partners[c:c + 4])))
    lines.append("END\n")

    with open(path, "w") as f:
        f.writelines(lines)
    return path


# ---------------------------------------------------------------------
# force-field XML
# ---------------------------------------------------------------------
def _xml_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def write_hybrid_ff_xml(structure, blueprint, path, hyb_resname="HYB") -> str:
    """Write a self-contained OpenMM-style forcefield XML for the hybrid residue(s). Returns path.

    Reads all force parameters from a plain OpenMM System built from the hybrid Structure, so the
    values are exact and already in OpenMM units (nm, kJ/mol, radians). Charge is per-ATOM (given in
    the <Residue>), LJ is per-TYPE (in <NonbondedForce>); bonded params are grouped by atom-type
    class as OpenMM does. NOTE: this is an INSPECTION dump -- a hybrid residue overlays several side
    chains on one core, so it is not a physically loadable molecule; it lets you check every number.
    """
    sites = _hyb_sites(structure, blueprint, hyb_resname)
    system = structure.createSystem(nonbondedMethod=app.NoCutoff)   # plain (unscaled) -> real params
    nbf = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    bond_forces = [f for f in system.getForces() if isinstance(f, mm.HarmonicBondForce)]
    angle_forces = [f for f in system.getForces() if isinstance(f, mm.HarmonicAngleForce)]
    tor_forces = [f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce)]
    hyb_global = set().union(*[s["atomset"] for s in sites]) if sites else set()

    def typ(i):
        return structure.atoms[i].type or f"X{structure.atoms[i].atomic_number}"

    _E, _NM, _KJ = unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole
    _RAD = unit.radian

    # per-type: element, mass, sigma, epsilon (LJ is one per synthesized type)
    types = OrderedDict()
    charge = {}
    for i in sorted(hyb_global):
        a = structure.atoms[i]
        q, sig, eps = nbf.getParticleParameters(i)
        charge[i] = q.value_in_unit(_E)
        t = typ(i)
        if t not in types:
            types[t] = dict(element=_element_symbol(a.atomic_number), mass=float(a.mass),
                            sigma=sig.value_in_unit(_NM), eps=eps.value_in_unit(_KJ))

    def touches(idxs):
        return any(i in hyb_global for i in idxs)

    # bonds grouped by class pair
    bonds = OrderedDict()
    for f in bond_forces:
        for b in range(f.getNumBonds()):
            p1, p2, length, k = f.getBondParameters(b)
            if not touches((p1, p2)):
                continue
            key = tuple(sorted((typ(p1), typ(p2))))
            bonds.setdefault(key, (length.value_in_unit(_NM), k.value_in_unit(_KJ / _NM ** 2)))

    # angles grouped by class triple (vertex = middle)
    angles = OrderedDict()
    for f in angle_forces:
        for a in range(f.getNumAngles()):
            p1, p2, p3, ang, k = f.getAngleParameters(a)
            if not touches((p1, p2, p3)):
                continue
            ends = sorted((typ(p1), typ(p3)))
            key = (ends[0], typ(p2), ends[1])
            angles.setdefault(key, (ang.value_in_unit(_RAD), k.value_in_unit(_KJ / _RAD ** 2)))

    # which 4-atom sets are impropers (ParmEd knows; the System's PeriodicTorsionForce does not)
    improper_quads = set()
    for d in getattr(structure, "dihedrals", []):
        if getattr(d, "improper", False):
            improper_quads.add(frozenset((d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx)))

    # torsions grouped by (class tuple, periodicity); label proper vs improper
    propers, impropers = OrderedDict(), OrderedDict()
    for f in tor_forces:
        for t in range(f.getNumTorsions()):
            p1, p2, p3, p4, per, phase, k = f.getTorsionParameters(t)
            if not touches((p1, p2, p3, p4)):
                continue
            classes = (typ(p1), typ(p2), typ(p3), typ(p4))
            entry = (int(per), phase.value_in_unit(_RAD), k.value_in_unit(_KJ))
            bucket = impropers if frozenset((p1, p2, p3, p4)) in improper_quads else propers
            bucket.setdefault(classes, []).append(entry)

    # ---- render ----
    L = ['<ForceField>']
    L.append('  <!-- MSLD hybrid residue(s): overlaid alchemical variants; inspection only, not a'
             ' physically loadable molecule. Charge is per-atom (Residue); LJ per-type. -->')
    L.append('  <AtomTypes>')
    for t, d in types.items():
        L.append('    <Type name="%s" class="%s" element="%s" mass="%.4f"/>'
                 % (_xml_escape(t), _xml_escape(t), d["element"], d["mass"]))
    L.append('  </AtomTypes>')

    L.append('  <Residues>')
    for s in sites:
        L.append('    <Residue name="%s">' % _xml_escape(s["resname"]))
        aset = s["atomset"]
        for i in s["atoms"]:
            L.append('      <Atom name="%s" type="%s" charge="%.6f"/>'
                     % (_xml_escape(structure.atoms[i].name), _xml_escape(typ(i)), charge[i]))
        seen_bonds = set()
        for b in structure.bonds:
            i, j = b.atom1.idx, b.atom2.idx
            if i in aset and j in aset and (i, j) not in seen_bonds:
                seen_bonds.add((i, j))
                L.append('      <Bond atomName1="%s" atomName2="%s"/>'
                         % (_xml_escape(structure.atoms[i].name), _xml_escape(structure.atoms[j].name)))
        for b in structure.bonds:                                  # bonds leaving the residue
            i, j = b.atom1.idx, b.atom2.idx
            ext = (i in aset) ^ (j in aset)
            if ext:
                inside = i if i in aset else j
                L.append('      <ExternalBond atomName="%s"/>' % _xml_escape(structure.atoms[inside].name))
        L.append('    </Residue>')
    L.append('  </Residues>')

    L.append('  <HarmonicBondForce>')
    for (c1, c2), (length, k) in bonds.items():
        L.append('    <Bond class1="%s" class2="%s" length="%.6f" k="%.3f"/>'
                 % (_xml_escape(c1), _xml_escape(c2), length, k))
    L.append('  </HarmonicBondForce>')

    L.append('  <HarmonicAngleForce>')
    for (c1, c2, c3), (ang, k) in angles.items():
        L.append('    <Angle class1="%s" class2="%s" class3="%s" angle="%.6f" k="%.3f"/>'
                 % (_xml_escape(c1), _xml_escape(c2), _xml_escape(c3), ang, k))
    L.append('  </HarmonicAngleForce>')

    L.append('  <PeriodicTorsionForce>')
    for tag, table in (("Proper", propers), ("Improper", impropers)):
        for classes, terms in table.items():
            attrs = 'class1="%s" class2="%s" class3="%s" class4="%s"' % tuple(_xml_escape(c) for c in classes)
            for n, (per, phase, k) in enumerate(terms, start=1):
                attrs += ' periodicity%d="%d" phase%d="%.6f" k%d="%.4f"' % (n, per, n, phase, n, k)
            L.append('    <%s %s/>' % (tag, attrs))
    L.append('  </PeriodicTorsionForce>')

    L.append('  <NonbondedForce coulomb14scale="0.8333333333333334" lj14scale="0.5">')
    for t, d in types.items():
        L.append('    <Atom type="%s" sigma="%.6f" epsilon="%.6f"/>'
                 % (_xml_escape(t), d["sigma"], d["eps"]))
    L.append('  </NonbondedForce>')
    L.append('</ForceField>')

    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path
