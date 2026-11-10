"""MCSMapper — discover the shared core of ligand variants by maximum common substructure.

Peptides share their core BY ATOM NAME (AminoAcidBackboneMapper). Arbitrary small molecules do
not, so the shared scaffold must be found by graph matching: the maximum common substructure
(MCS) across all variants. Done with RDKit's `rdFMCS`, so MCSMapper uses it as the
backend.

Flow (per site):
    reference molecule ─┐
    variant molecules  ─┴─ rdFMCS ─▶ per-variant {variant atom ↔ reference atom} on the MCS
                                     ─▶ core = reference atoms matched in every variant
                                        (optionally filtered to atoms sharing an LJ TYPE, §16.4)
                                     ─▶ build_hybrid_local_mapping  ─▶ CoreMapping
                                     ─▶ ring-break guard: reject a map that covers a proper,
                                        non-empty subset of a ring (MSLD assumes a PENDANT change)

Molecules are read as graphs from any structure whose atoms expose `.atomic_number` and whose
`.bonds` give connectivity (a ParmEd Structure, or any equivalent).
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .core import Mapper, CoreMapping, build_hybrid_local_mapping


def molecular_graph(structure):
    """(adjacency, elements) for a structure: adjacency[i] = set of bonded atom indices."""
    n = len(structure.atoms)
    adjacency = [set() for _ in range(n)]
    for b in structure.bonds:
        adjacency[b.atom1.idx].add(b.atom2.idx)
        adjacency[b.atom2.idx].add(b.atom1.idx)
    elements = [int(a.atomic_number) for a in structure.atoms]
    return adjacency, elements


def _rings(adjacency) -> list[set[int]]:
    """Atoms that lie on some cycle, grouped by simple ring (SSSR-ish). Cheap heuristic: an atom
    is in a ring iff removing a tree still leaves it with degree >= 2 in a biconnected sense; we
    approximate with 'atoms not on any bridge'. Good enough for the ring-break guard."""
    # find bridges (edges whose removal disconnects) via DFS low-link; ring atoms touch a non-bridge
    n = len(adjacency)
    disc = [-1] * n
    low = [0] * n
    timer = [0]
    ring_atoms: set[int] = set()

    def dfs(u, parent):
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        for v in adjacency[u]:
            if v == parent:
                continue
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] <= disc[u]:            # (u,v) is NOT a bridge -> both on a cycle
                    ring_atoms.add(u)
                    ring_atoms.add(v)
            else:
                low[u] = min(low[u], disc[v])
                if disc[v] < disc[u]:
                    ring_atoms.add(u)
                    ring_atoms.add(v)

    for s in range(n):
        if disc[s] == -1:
            dfs(s, -1)
    return ring_atoms


@dataclass
class MCSMapper(Mapper):
    """Graph-MCS mapper for ligand variants (RDKit backend).

    reference_name    : block label for the reference molecule (defines the env frame).
    include_reference : the reference also competes as a block (default True, per [D3]).
    match_types       : an MCS-matched atom joins the env only if its atom TYPE agrees across
                        variants (env is unscaled -> one LJ type; §16.4). Otherwise it is a block.
    reject_ring_break : refuse a mapping that covers only PART of a ring (MSLD assumes a pendant
                        change, not a ring edit).
    ring_matches_ring_only / complete_rings_only : passed to rdFMCS.
    """
    reference_name: str = "REF"
    include_reference: bool = True
    match_types: bool = True
    reject_ring_break: bool = True
    ring_matches_ring_only: bool = True
    complete_rings_only: bool = True

    def get_mapping(self, reference=None, variants=None) -> CoreMapping:
        if reference is None or not variants:
            raise ValueError("MCSMapper needs a reference structure and >= 1 variant")

        ref_mol = _to_rdkit_mol(reference)
        # per variant: {variant local idx -> reference local idx} on the pairwise MCS
        corr_to_ref: dict[str, dict[int, int]] = {}
        for name, vstruct in variants.items():
            vmol = _to_rdkit_mol(vstruct)
            corr_to_ref[name] = _pairwise_correspondence(ref_mol, vmol, self)

        # core = reference atoms matched in EVERY variant (and, if match_types, one shared type)
        matched_ref_per_variant = [set(c.values()) for c in corr_to_ref.values()]
        core_ref = set.intersection(*matched_ref_per_variant) if matched_ref_per_variant else set()
        if self.match_types:
            core_ref = {rl for rl in core_ref
                        if _type_consistent(rl, reference, variants, corr_to_ref)}
        core_ref_locals = sorted(core_ref)

        # ring-break guard: no core that is a proper, non-empty subset of any variant's ring
        if self.reject_ring_break:
            _assert_no_ring_break(reference, variants, core_ref_locals, corr_to_ref,
                                  self.reference_name, self.include_reference)

        # assemble the hybrid-local CoreMapping (shared with the peptide mapper)
        slot_of_ref = {rl: k for k, rl in enumerate(core_ref_locals)}
        blocks_struct: dict[str, object] = {}
        per_block_slot: dict[str, dict[int, int]] = {}
        if self.include_reference:
            blocks_struct[self.reference_name] = reference
            per_block_slot[self.reference_name] = {rl: slot_of_ref[rl] for rl in core_ref_locals}
        for name, vstruct in variants.items():
            blocks_struct[name] = vstruct
            inv = {vl: rl for vl, rl in corr_to_ref[name].items() if rl in slot_of_ref}
            per_block_slot[name] = {vl: slot_of_ref[rl] for vl, rl in inv.items()}
        if len(blocks_struct) < 2:
            raise ValueError("a site needs >= 2 competing variants (or include_reference=True)")

        core_slot_names = [reference.atoms[rl].name for rl in core_ref_locals]
        metadata = {"reference": self.reference_name, "n_core": len(core_ref_locals),
                    "method": "mcs"}
        return build_hybrid_local_mapping(blocks_struct, per_block_slot, core_slot_names,
                                          metadata=metadata)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _to_rdkit_mol(structure):
    """Build an RDKit Mol from a structure's elements + bonds (single bonds; no sanitize).

    We deliberately do NOT sanitize: our element-only, all-single-bond graphs can have
    non-physical valences that sanitization would reject. But rdFMCS and GetSubstructMatch need
    ring perception, so initialize the property cache (non-strict) + ring info explicitly.
    """
    rw = Chem.RWMol()
    for a in structure.atoms:
        rw.AddAtom(Chem.Atom(int(a.atomic_number)))
    for b in structure.bonds:
        rw.AddBond(b.atom1.idx, b.atom2.idx, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def _pairwise_correspondence(ref_mol, var_mol, cfg) -> dict[int, int]:
    """{var atom idx -> ref atom idx} for the MCS of (ref, var). Empty if no MCS."""
    res = rdFMCS.FindMCS(
        [ref_mol, var_mol],
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareAny,
        ringMatchesRingOnly=cfg.ring_matches_ring_only,
        completeRingsOnly=cfg.complete_rings_only,
        timeout=10,
    )
    if not res.smartsString:
        return {}
    query = Chem.MolFromSmarts(res.smartsString)
    ref_match = ref_mol.GetSubstructMatch(query)
    var_match = var_mol.GetSubstructMatch(query)
    if not ref_match or not var_match:
        return {}
    return {var_match[i]: ref_match[i] for i in range(len(ref_match))}


def _type_consistent(ref_local, reference, variants, corr_to_ref) -> bool:
    """True if the atom type at ref_local agrees across the reference and every variant."""
    ref_type = reference.atoms[ref_local].type
    for name, vstruct in variants.items():
        inv = {rl: vl for vl, rl in corr_to_ref[name].items()}
        vl = inv.get(ref_local)
        if vl is None or vstruct.atoms[vl].type != ref_type:
            return False
    return True


def _assert_no_ring_break(reference, variants, core_ref_locals, corr_to_ref,
                          reference_name, include_reference) -> None:
    core_ref = set(core_ref_locals)
    checks = []
    if include_reference:
        checks.append((reference_name, reference, {rl: rl for rl in core_ref}))
    for name, vstruct in variants.items():
        inv = {rl: vl for vl, rl in corr_to_ref[name].items()}
        checks.append((name, vstruct, {rl: inv[rl] for rl in core_ref if rl in inv}))
    for name, struct, ref_to_local in checks:
        adjacency, _ = molecular_graph(struct)
        ring_atoms = _rings(adjacency)
        core_locals = set(ref_to_local.values())
        # group ring atoms into connected ring systems, check each is fully in or out of core
        seen: set[int] = set()
        for start in ring_atoms:
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                u = stack.pop()
                if u in comp:
                    continue
                comp.add(u)
                seen.add(u)
                for v in adjacency[u]:
                    if v in ring_atoms and v not in comp:
                        stack.append(v)
            inside = comp & core_locals
            if inside and inside != comp:
                raise ValueError(
                    f"MCS map breaks a ring in variant '{name}': ring atoms {sorted(comp)} are "
                    f"only partly in the shared core {sorted(inside)}. MSLD assumes a pendant "
                    f"change, not a ring edit; set reject_ring_break=False to override.")
