"""hybrid_output.py — the merged-residue PDB (+ CONECT) and the self-contained FF XML.

Builds a real hybrid (MDM2 peptide, ALA4 site, SER variant competing with WT), writes both
artifacts, and checks the things the raw ParmEd topology got wrong:
  - the site is ONE residue (HYB) holding the shared core + every variant's side chain,
  - contiguous, on the site's chain (not chain-less), with no extra stacked residues,
  - CONECT records are written (a novel residue),
  - the FF XML is well-formed and describes exactly that residue.
"""
import copy
import os
import sys
import tempfile
import types
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import openmm as mm
import parmed as pmd
from openmm import app, unit

from openmm_ld.prep import build_msld_system, Site
from openmm_ld.prep.assemble.hybrid_output import (write_hybrid_pdb, write_hybrid_ff_xml,
                                              _unique4, _is_solvent_res)

_REF = mm.Platform.getPlatform("Reference")
_PDB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "examples", "mdm2", "mdm2_peptide.pdb")
_cache = {}


def _bundle():
    if "b" not in _cache:
        warnings.simplefilter("ignore")
        pdb = app.PDBFile(_PDB)
        ff = app.ForceField("amber14-all.xml")
        mod = app.Modeller(pdb.topology, pdb.positions)
        mod.addHydrogens(ff)
        sysm = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff)
        xyz = np.array(mod.positions.value_in_unit(unit.nanometer)) * 10.0
        st = pmd.openmm.load_topology(mod.topology, system=sysm, xyz=xyz)
        for r in st.residues:
            r.chain = "B"                                     # give the peptide a real chain
        sub = copy.copy(st)                                   # SER side chain donor (residue 1)
        keep = {a.idx for a in list(st.residues)[1].atoms}
        sub.strip(np.array([i not in keep for i in range(len(st.atoms))]))
        bundle = build_msld_system(
            environment=st,
            sites=[Site(residue=3, variants={"SER": sub}, mapper="backbone")],  # ALA4
            renormalize=True, solvate=False)
        _cache["b"] = bundle
    return _cache["b"]


def _atom_lines(pdb_text):
    return [l for l in pdb_text.splitlines() if l.startswith(("ATOM", "HETATM"))]


def test_site_becomes_one_merged_residue_on_the_chain():
    bundle = _bundle()
    path = os.path.join(tempfile.mkdtemp(), "hyb.pdb")
    write_hybrid_pdb(bundle.structure, bundle.mapping, path)
    text = open(path).read()
    lines = _atom_lines(text)

    hyb = [l for l in lines if l[17:20].strip() == "HYB"]
    assert hyb, "no HYB residue written"

    # exactly one HYB residue: single (chain, resSeq)
    assert len({(l[21], l[22:26]) for l in hyb}) == 1, "HYB split across residues"
    assert {l[21] for l in hyb} == {"B"}, "HYB not on the site chain"

    # it holds the shared core AND the SER-specific side chain (OG), in one residue
    names = [l[12:16].strip() for l in hyb]
    assert {"N", "CA", "C", "O", "CB"} <= set(names)          # shared backbone core
    assert "OG" in names                                       # SER block atom merged in
    assert len(names) == len(set(names)), "duplicate atom names in HYB"

    # contiguous block of lines (no stray stacked residues interleaved)
    idx = [i for i, l in enumerate(lines) if l[17:20].strip() == "HYB"]
    assert idx == list(range(idx[0], idx[0] + len(idx))), "HYB atoms not contiguous"

    # one residue per (chain, resSeq, resName) == the env's residue count (no appended residues)
    n_res = len({(l[17:20], l[21], l[22:26]) for l in lines})
    assert n_res == len(bundle.structure.residues) - len(bundle.mapping.sites[0].blocks) + 1, n_res

    assert "CONECT" in text                                    # bonds spelled out for the novel res


def test_ff_xml_is_well_formed_and_describes_the_residue():
    bundle = _bundle()
    path = os.path.join(tempfile.mkdtemp(), "hyb.xml")
    write_hybrid_ff_xml(bundle.structure, bundle.mapping, path)
    root = ET.parse(path).getroot()                            # must be valid XML
    assert root.tag == "ForceField"

    res = root.findall("./Residues/Residue")
    assert len(res) == 1 and res[0].get("name") == "HYB"
    atoms = res[0].findall("Atom")
    site = bundle.mapping.sites[0]
    n_site = len(site.env_atoms) + len(site.all_block_atoms())
    assert len(atoms) == n_site                                # every site atom present, once
    assert all(a.get("type") and a.get("charge") is not None for a in atoms)
    assert res[0].findall("Bond") and res[0].findall("ExternalBond")   # intra + backbone bonds

    # the sections that let you check the numbers exist and are populated
    assert root.findall("./AtomTypes/Type")
    assert root.findall("./HarmonicBondForce/Bond")
    assert root.findall("./HarmonicAngleForce/Angle")
    assert root.findall("./NonbondedForce/Atom")
    # every atom's type is defined in <AtomTypes>
    declared = {t.get("name") for t in root.findall("./AtomTypes/Type")}
    assert {a.get("type") for a in atoms} <= declared


def test_unique4_avoids_truncation_collisions():
    # overlaid variants disambiguate by appending a digit (HD11 -> HD111); truncated to the 4-char
    # PDB field both become "HD11". _unique4 must keep them distinct within 4 chars.
    used = set()
    a = _unique4("HD11", used); used.add(a)
    b = _unique4("HD111", used); used.add(b)
    c = _unique4("HD121", used); used.add(c)
    assert a == "HD11"
    assert b != a and len(b) <= 4
    assert c not in (a, b) and len(c) <= 4
    assert len({a, b, c}) == 3


def _mini_two_molecule_structure():
    """Two 2-atom residues in ONE chain A, NOT bonded to each other, with a periodic box."""
    st = pmd.Structure()
    spec = [(0.0, "ALA", 1, "N"), (1.5, "ALA", 1, "CA"),
            (30.0, "GLY", 2, "N"), (31.5, "GLY", 2, "CA")]      # residue 2 is 30 A away, no bond
    for x, resn, num, nm in spec:
        at = pmd.Atom(name=nm, atomic_number=7 if nm == "N" else 6)
        at.xx, at.xy, at.xz = x, 0.0, 0.0
        st.add_atom(at, resn, num, "A")
    st.bonds.append(pmd.Bond(st.atoms[0], st.atoms[1]))         # intra-residue bonds only
    st.bonds.append(pmd.Bond(st.atoms[2], st.atoms[3]))
    st.box = [50.0, 50.0, 50.0, 90.0, 90.0, 90.0]
    return st


def test_cryst1_and_ter_at_real_chain_break():
    st = _mini_two_molecule_structure()
    path = os.path.join(tempfile.mkdtemp(), "mini.pdb")
    write_hybrid_pdb(st, types.SimpleNamespace(sites=[]), path)   # no sites -> plain write path
    text = open(path).read()

    # 1) periodic box is emitted
    assert text.startswith("CRYST1"), "CRYST1 not written for a boxed structure"
    assert "50.000" in text.splitlines()[0]

    # 2) a TER separates the two residues even though they share chain A (they are NOT bonded),
    #    so a viewer will not draw a bond across the 30 A gap
    lines = text.splitlines()
    ala = next(i for i, l in enumerate(lines) if l[17:20].strip() == "ALA")
    gly = next(i for i, l in enumerate(lines) if l[17:20].strip() == "GLY")
    assert any(lines[i].startswith("TER") for i in range(ala, gly)), "no TER at the chain break"


def test_no_cryst1_without_a_box():
    st = _mini_two_molecule_structure()
    st.box = None
    path = os.path.join(tempfile.mkdtemp(), "nobox.pdb")
    write_hybrid_pdb(st, types.SimpleNamespace(sites=[]), path)
    assert not open(path).read().startswith("CRYST1")
    assert _is_solvent_res(list(st.residues)[0]) is False        # 2-atom residue is not "solvent"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("\nhybrid output tests passed")
