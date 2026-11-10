"""DataIngestPipeline — turn input files into a parametrized ParmEd Structure.

Whatever the user provides, ingest emits one common representation: a ParmEd
`Structure` carrying chemistry and force-field parameters (types, charges, bonded terms, LJ,
NBFIX).

    MODE B  parametrized system (self-contained, ParmEd-native)
      AmberInput   prmtop (+ inpcrd)            -> load_amber
      CharmmInput  psf + rtf/prm(/str) + coords -> load_charmm   (psf is not self-contained:
                                                                  it needs the parameter set)
    MODE A  apply a force field to a PDB (needs a template for every residue)
      ForceFieldInput  pdb + ffxml(s)           -> apply_forcefield  (+ recover NBFIX recover_charmm_lj)
"""
from __future__ import annotations

import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field

import numpy as np
import parmed as pmd
from openmm import app, unit, Vec3
from rdkit import Chem

from .recover import recover_charmm_lj

_NM = unit.nanometer


def _fspath(p):
    """Convert a path-like (str or pathlib.Path) to str (OpenMM's PDBFile/RDKit only accept str)"""
    return os.fspath(p) if p is not None else None


# ---------------------------------------------------------------------
# Source specs 
# ---------------------------------------------------------------------
@dataclass
class AmberInput:
    """Mode B: a self-contained Amber prmtop (+ optional inpcrd/rst7 for coordinates)."""
    prmtop: str
    coordinates: str | None = None


@dataclass
class CharmmInput:
    """Mode B: a CHARMM psf + its parameter files (rtf/prm/str) + a coordinate file (pdb/crd).

    A psf is NOT self-contained -- it carries atom types/charges/connectivity but not the force
    constants or LJ, so the parameter set is required.
    """
    psf: str
    parameters: list[str] = field(default_factory=list)
    coordinates: str | None = None


@dataclass
class ForceFieldInput:
    """Mode A: apply OpenMM force field(s) to a structure file.

    SDF is accepted ONLY for ligands (parametrized by a small-molecule force field, 'gaff-2.11' or
    'openff-2.2.0'). Amino acids (standard or non-standard) and other biopolymers must be a PDB
    with a standard ffxml (amber14-all.xml, charmm36.xml, or a residue-specific ffxml). See
    apply_forcefield for the accepted combinations.

    ignore_external_bonds : match residue templates as if the structure had no external bonds. 
    """
    structure: str
    forcefield: list[str] = field(default_factory=list)
    add_hydrogens: bool = False
    ignore_external_bonds: bool = False


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_amber(prmtop: str, coordinates: str | None = None) -> pmd.Structure:
    """Mode B Amber: read a prmtop (self-contained) + optional coordinate file."""
    prmtop, coordinates = _fspath(prmtop), _fspath(coordinates)
    st = pmd.load_file(prmtop, xyz=coordinates) if coordinates else pmd.load_file(prmtop)
    return st


def load_charmm(psf: str, parameters, coordinates: str | None = None) -> pmd.Structure:
    """Mode B CHARMM: psf + parameter set (rtf/prm/str) + optional coordinates (pdb/crd).

    NBFIX flows through natively (ParmEd builds the tabulated LJ from the atom types on
    createSystem), so no recovery is needed on this path.
    """
    psf, coordinates = _fspath(psf), _fspath(coordinates)
    parameters = [_fspath(p) for p in parameters]
    st = pmd.charmm.CharmmPsfFile(psf)
    st.load_parameters(pmd.charmm.CharmmParameterSet(*parameters))
    if coordinates:
        crd = pmd.load_file(coordinates)
        st.coordinates = crd.coordinates
    return st


def _sdf_to_openmm(sdf: str):
    """Read the first molecule of an SDF into an OpenMM (Topology, positions) using RDKit.

    SDF carries elements, 3-D coordinates, and bonds, but NO force-field parameters (those come
    from the ffxml at createSystem). Atoms are named <element><n> (SDF has no atom names) and the
    residue takes the SDF molecule title. Explicit hydrogens in the SDF are kept.
    """
    mol = next((m for m in Chem.SDMolSupplier(_fspath(sdf), removeHs=False) if m is not None), None)
    if mol is None:
        raise ValueError(f"could not read a molecule from SDF {sdf!r}")
    if mol.GetNumConformers() == 0:
        raise ValueError(f"SDF {sdf!r} has no 3-D coordinates")
    top = app.Topology()
    resname = (mol.GetProp("_Name").strip() if mol.HasProp("_Name") else "") or "LIG"
    res = top.addResidue(resname, top.addChain())
    atoms, counts = [], {}
    for a in mol.GetAtoms():
        el = app.Element.getByAtomicNumber(a.GetAtomicNum())
        counts[el.symbol] = counts.get(el.symbol, 0) + 1
        atoms.append(top.addAtom(f"{el.symbol}{counts[el.symbol]}", el, res))
    for b in mol.GetBonds():
        top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    conf = mol.GetConformer()
    pos = [Vec3(conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z)
           for i in range(mol.GetNumAtoms())] * unit.angstrom
    return top, pos


_SMALLMOL_FF_PREFIXES = ("gaff", "openff", "smirnoff")

def _is_smallmol_ff(name: str) -> bool:
    """True for a small-molecule FF spec (gaff*/openff*/smirnoff*), False for a standard ffxml."""
    return name.lower().rsplit("/", 1)[-1].startswith(_SMALLMOL_FF_PREFIXES)


def _register_smallmol_ff(ff, sdf: str, ff_name: str) -> None:
    """Register a GAFF/OpenFF residue-template generator for the SDF's molecule on `ff`.

    openff.toolkit + openmmforcefields are imported here, not at module top: they pull a heavy stack
    (~1.5 s incl. pytorch) needed only on this SDF-parametrization path, so paying it on every
    `import openmm_ld.prep` is not worth it (same rationale as the deferred import in model.py). They are
    hard deps in env/environment.yml -- this is a performance deferral, not an optional-dep fallback.
    """
    # GAFF shells out to AmberTools `antechamber`, which lives beside the interpreter in a conda env
    # but is only on PATH when the env is activated; make it findable either way.
    if shutil.which("antechamber") is None:
        bindir = os.path.join(sys.prefix, "bin")
        if os.path.exists(os.path.join(bindir, "antechamber")):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")

    from openff.toolkit import Molecule
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
    molecule = Molecule.from_file(sdf)
    if isinstance(molecule, list):
        molecule = molecule[0]                      # first record of a multi-molecule SDF
    Gen = (GAFFTemplateGenerator if ff_name.lower().rsplit("/", 1)[-1].startswith("gaff")
           else SMIRNOFFTemplateGenerator)
    ff.registerTemplateGenerator(Gen(molecules=molecule, forcefield=ff_name).generator)


def apply_forcefield(structure: str, forcefield, *, add_hydrogens: bool = False,
                     ignore_external_bonds: bool = False) -> pmd.Structure:
    """Mode A: parametrize a structure file with OpenMM ForceField(s) -> ParmEd Structure.

    Two accepted combinations (SDF is for LIGANDS ONLY):
      * a PDB + standard ffxml(s) (amber14-all.xml, charmm36.xml, a file path) -- proteins, nucleic
        acids, and standard OR non-standard amino acids (the latter with a matching residue ffxml);
      * a ligand SDF + ONE small-molecule force field -- 'gaff-2.11' (GAFF) or 'openff-2.2.0'
        (OpenFF/SMIRNOFF) -- which generates the ligand's template from the SDF molecule.
    An SDF without a small-molecule FF, or a small-molecule FF without an SDF, is refused. Recovers
    CHARMM NBFIX.

    ignore_external_bonds : passed to createSystem so a MONOMER fragment (e.g. a lone amino-acid
        variant residue) matches its in-chain template despite missing the peptide-bond partners at
        N and C. Yields the correct in-chain types/charges with no terminal OXT/H (verified: charges
        identical to the same residue in a full chain). See ForceFieldInput.
    """
    structure = _fspath(structure)
    forcefield = [_fspath(f) for f in forcefield]
    smallmol = [f for f in forcefield if _is_smallmol_ff(f)]
    ff = app.ForceField(*[f for f in forcefield if not _is_smallmol_ff(f)])
    is_sdf = os.path.splitext(structure)[1].lower() == ".sdf"

    # POLICY: SDF <-> ligand force field, both directions. SDF is accepted only for ligands;
    # amino acids (standard or non-standard) must be provided as PDB with a standard ffxml.
    if is_sdf and not smallmol:
        raise ValueError(
            "an SDF is accepted only for ligands and needs a small-molecule force field "
            "('gaff-2.11' or 'openff-2.2.0'); got a standard force field. Provide amino acids "
            "(standard or non-standard) as a PDB with a standard force field (e.g. amber14-all.xml).")
    if smallmol and not is_sdf:
        raise ValueError(
            "a small-molecule (ligand) force field (gaff*/openff*) needs an SDF structure; a PDB "
            "does not carry bond orders.")
    if len(smallmol) > 1:
        raise ValueError(f"pass at most one small-molecule force field; got {smallmol}")
    if smallmol:
        _register_smallmol_ff(ff, structure, smallmol[0])

    if is_sdf:
        top, pos = _sdf_to_openmm(structure)
    else:
        pdbfile = app.PDBFile(structure)
        top, pos = pdbfile.topology, pdbfile.positions
    if add_hydrogens:
        mod = app.Modeller(top, pos)
        mod.addHydrogens(ff)
        top, pos = mod.topology, mod.positions
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff,
                             ignoreExternalBonds=ignore_external_bonds)
    xyz = np.array(pos.value_in_unit(_NM)) * 10.0
    with warnings.catch_warnings():
        # both are expected on the OpenMM->ParmEd hop and do not affect the parameters
        # (verified: energy matches a direct createSystem to ~1e-10 relative)
        warnings.filterwarnings("ignore", message="Unsupported Force type")
        warnings.filterwarnings("ignore", message="Detected incomplete exceptions")
        st = pmd.openmm.load_topology(top, system=system, xyz=xyz)
    recover_charmm_lj(st, system)                    # CHARMM: restore dropped LJ; no-op on Amber
    return st


# ---------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------
class DataIngestPipeline:
    """Resolve a source spec to a parametrized ParmEd Structure, validating sufficiency."""

    def load(self, source) -> pmd.Structure:
        if isinstance(source, AmberInput):
            self._require_file(source.prmtop, "Amber prmtop")
            if source.coordinates:
                self._require_file(source.coordinates, "Amber coordinates")
            return load_amber(source.prmtop, source.coordinates)

        if isinstance(source, CharmmInput):
            if not source.parameters:                # spec completeness before file existence
                raise ValueError(
                    "CharmmInput needs >= 1 parameter file (rtf/prm/str): a psf is not "
                    "self-contained (it has no force constants or LJ).")
            self._require_file(source.psf, "CHARMM psf")
            for p in source.parameters:
                self._require_file(p, "CHARMM parameter")
            if source.coordinates:
                self._require_file(source.coordinates, "CHARMM coordinates")
            return load_charmm(source.psf, source.parameters, source.coordinates)

        if isinstance(source, ForceFieldInput):
            self._require_file(source.structure, "structure (PDB/SDF)")
            if not source.forcefield:
                raise ValueError(
                    "ForceFieldInput needs >= 1 force field (an ffxml name like "
                    "'amber14-all.xml' or a file path).")
            return apply_forcefield(source.structure, source.forcefield,
                                    add_hydrogens=source.add_hydrogens,
                                    ignore_external_bonds=source.ignore_external_bonds)

        raise TypeError(
            f"unknown ingest source {type(source).__name__}; use AmberInput, CharmmInput, "
            f"or ForceFieldInput")

    def ingest(self, *paths, forcefield=None, add_hydrogens: bool = False,
               coordinates=None, ignore_external_bonds: bool = False) -> pmd.Structure:
        """One door: infer the route from the file types, then load (see resolve_source)."""
        return self.load(resolve_source(
            *paths, forcefield=forcefield, add_hydrogens=add_hydrogens,
            coordinates=coordinates, ignore_external_bonds=ignore_external_bonds))

    @staticmethod
    def _require_file(path: str, what: str) -> None:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"{what} file not found: {path!r}")


def load_structure(source) -> pmd.Structure:
    """Convenience: DataIngestPipeline().load(source)."""
    return DataIngestPipeline().load(source)


# ---------------------------------------------------------------------
# One door: infer the route from the file types 
# ---------------------------------------------------------------------
# The user should not have to know Mode A vs Mode B, or Amber vs CHARMM. `ingest(*paths)`
# inspects the file extensions and builds the right source spec, then loads it. 
# Chemistry decisions are NOT guessed here: `forcefield=` stays an explicit argument 
# because ffxml names (e.g. 'amber14-all.xml') are OpenMM resource names, not
# files on disk, so they cannot be sniffed from the inputs.

_AMBER_TOP = {".prmtop", ".parm7", ".top7"}
_CHARMM_PSF = {".psf"}
_CHARMM_PRM = {".prm", ".par", ".rtf", ".str"}
_COORDS = {".inpcrd", ".rst7", ".restrt", ".ncrst", ".crd", ".coor", ".rst"}
_PDB = {".pdb"}
_SDF = {".sdf"}


def _ext_kind(path: str) -> str:
    """Classify one path by its extension into a routing bucket."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _AMBER_TOP:  return "amber_top"
    if ext in _CHARMM_PSF: return "psf"
    if ext in _CHARMM_PRM: return "charmm_prm"
    if ext in _COORDS:     return "coords"
    if ext in _PDB:        return "pdb"
    if ext in _SDF:        return "sdf"
    return "unknown"


def resolve_source(*paths, forcefield=None, add_hydrogens: bool = False, coordinates=None,
                   ignore_external_bonds: bool = False):
    """Inspect file extensions and return the matching source spec.

    Routing:
        *.prmtop/.parm7                 -> AmberInput      (Mode B)
        *.psf  + *.prm/.rtf/.str        -> CharmmInput     (Mode B)
        *.pdb/.sdf  + forcefield=[...]  -> ForceFieldInput (Mode A)
    Coordinate files (*.inpcrd/.rst7/.crd/..., or a .pdb) attach to a Mode-B topology; an explicit
    `coordinates=` wins. `ignore_external_bonds` is carried onto a ForceFieldInput (Mode A only);
    it is meaningless for a parametrized topology and ignored there. Raises a precise ValueError on
    an ambiguous or insufficient set. File existance is not checked here (DataIngestPipeline.load
    does that) - this only decides the route.
    """
    if not paths:
        raise ValueError("ingest needs at least one file path")

    kinds: dict[str, list] = {}
    for p in paths:
        kinds.setdefault(_ext_kind(p), []).append(p)
    if kinds.get("unknown"):
        raise ValueError(
            f"unrecognized file type(s): {kinds['unknown']}. Recognized: .prmtop/.parm7 "
            f"(Amber), .psf + .prm/.rtf/.str (CHARMM), .pdb/.sdf (+ forcefield=).")

    amber = kinds.get("amber_top", [])
    psf = kinds.get("psf", [])
    prm = kinds.get("charmm_prm", [])
    pdb = kinds.get("pdb", [])
    sdf = kinds.get("sdf", [])
    crd = kinds.get("coords", [])
    structure = pdb + sdf              # candidates for the ffxml route (Mode A)

    if amber and psf:
        raise ValueError(
            "both an Amber topology and a CHARMM psf were given; ingest one system at a time.")
    if sdf and (amber or psf):
        raise ValueError(
            "an .sdf is a structure for the ffxml route (Mode A); it cannot accompany a "
            "parametrized topology (.prmtop / .psf).")
    if len(amber) > 1:
        raise ValueError(f"multiple Amber topologies given: {amber}")
    if len(psf) > 1:
        raise ValueError(f"multiple CHARMM psf files given: {psf}")

    def _coords():
        if coordinates:
            return coordinates
        if crd:
            return crd[0]
        if pdb:
            return pdb[0]        # a .pdb can carry coordinates for a Mode-B topology
        return None

    if amber:
        if forcefield:
            raise ValueError(
                "forcefield= applies only to a structure file (.pdb/.sdf, Mode A); you also gave "
                "an Amber prmtop (already parametrized).")
        return AmberInput(prmtop=amber[0], coordinates=_coords())

    if psf:
        if not prm:
            raise ValueError(
                "a CHARMM .psf needs >= 1 parameter file (.prm/.rtf/.str): a psf carries atom "
                "types and connectivity but no force constants or LJ.")
        if forcefield:
            raise ValueError(
                "forcefield= applies only to a structure file (.pdb/.sdf, Mode A); you also gave "
                "a CHARMM psf.")
        return CharmmInput(psf=psf[0], parameters=prm, coordinates=_coords())

    if structure:
        if len(structure) > 1:
            raise ValueError(
                f"multiple structure files (.pdb/.sdf) and no topology to disambiguate: {structure}")
        if not forcefield:
            raise ValueError(
                "a lone structure (.pdb/.sdf) needs a force field to parametrize it. Pass "
                "forcefield=['amber14-all.xml', ...] (Mode A), or provide an already "
                "parametrized topology (.prmtop, or .psf + parameters).")
        return ForceFieldInput(structure=structure[0], forcefield=list(forcefield),
                               add_hydrogens=add_hydrogens,
                               ignore_external_bonds=ignore_external_bonds)

    raise ValueError(
        f"could not determine a topology from {list(paths)}; expected a .prmtop, a .psf "
        f"(+ parameters), or a .pdb/.sdf (+ forcefield=).")


def ingest(*paths, forcefield=None, add_hydrogens: bool = False,
           coordinates=None, ignore_external_bonds: bool = False) -> pmd.Structure:
    """One door: infer the route from the files and return a parametrized ParmEd Structure.

        ingest("system.prmtop", "system.inpcrd")               # Amber  (Mode B)
        ingest("mol.psf", "par_all36.prm", "mol.crd")          # CHARMM (Mode B)
        ingest("peptide.pdb", forcefield=["amber14-all.xml"])  # Mode A
        ingest("val_monomer.pdb", forcefield=["amber14-all.xml"],
               ignore_external_bonds=True)                     # Mode A, a lone residue fragment

    Equivalent to DataIngestPipeline().load(resolve_source(...)). See resolve_source for the
    routing rules and error cases.
    """
    return DataIngestPipeline().load(
        resolve_source(*paths, forcefield=forcefield, add_hydrogens=add_hydrogens,
                       coordinates=coordinates, ignore_external_bonds=ignore_external_bonds))
