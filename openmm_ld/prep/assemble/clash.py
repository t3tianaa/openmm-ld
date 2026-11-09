"""Clash check + rotamer relief for grafted blocks.

The structure builder places every variant's block by backbone alignment, in the rotamer the
variant file happens to have. Nothing checks that this rotamer fits the environment: on MDM2 the
grafted ILE put CD1 0.147 nm from MET38 SD (ILE endpoint ~3e5 kJ/mol above the others), and the
grafted SER put HB22 0.123 nm from a backbone H. At partial lambda the soft-core hides such a
clash, so minimization and dynamics run straight into a wall that keeps that lambda at 0.

  block_clashes    : per block, the non-bonded contacts with the environment (and inside the
                     block) that are closer than CONTACT_SCALE * (r_vdw_i + r_vdw_j).
  relieve_clashes  : for a clashing block, try the staggered rotamers of its chi1 (rotate the
                     whole block about the env attachment bond) and of each first-shell bond
                     inside the block, and keep the combination with the lowest clash score.
                     Bond lengths and angles are unchanged (rigid rotations about bonds).

Distances are geometric only (no force field): pairs within three bonds are skipped (1-2/1-3/1-4
geometry is fixed by the bonded terms), and a polar H next to an N/O acceptor gets a hydrogen-bond
contact distance instead of the van der Waals one. Coordinates are ParmEd's (angstrom); every
distance here is in nm.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

# van der Waals radii (nm, Bondi) by atomic number; unknown elements use carbon's
_VDW = {1: 0.110, 6: 0.170, 7: 0.155, 8: 0.152, 9: 0.147, 15: 0.180, 16: 0.180, 17: 0.175}
CONTACT_SCALE = 0.75        # clash if d < 0.75 * (r_i + r_j): H-H 0.165, C-C 0.255, C-S 0.263 nm
HBOND_CONTACT = 0.150       # polar H ... N/O: shorter than vdW is normal (H-bond, ~0.18-0.20 nm)
WARN_OVERLAP = 0.03         # nm; a clash deeper than this survives the rotamer search -> warn
_SEARCH_RADIUS = 0.30       # nm; no contact distance above exceeds this
_STAGGER = (0.0, 120.0, 240.0)


@dataclass
class Contact:
    atom: int                 # block atom
    partner: int              # environment (or same-block) atom
    distance: float           # nm
    contact: float            # nm, the clash threshold for this pair

    @property
    def overlap(self) -> float:
        return max(0.0, self.contact - self.distance)


@dataclass
class BlockClashReport:
    site: int                 # 1-based site index
    block: str
    clashes: list = field(default_factory=list)       # Contact, worst first
    score_before: float = 0.0
    score_after: float = 0.0
    rotation: dict = field(default_factory=dict)      # {(axis_from, axis_to): degrees} applied

    @property
    def worst(self):
        return self.clashes[0] if self.clashes else None

    @property
    def severe(self) -> bool:
        """a clash deeper than WARN_OVERLAP (shallow H...H touches are left to minimization)"""
        return any(c.overlap > WARN_OVERLAP for c in self.clashes)


def _coords_nm(structure) -> np.ndarray:
    return np.array([[a.xx, a.xy, a.xz] for a in structure.atoms]) / 10.0


def _graph(structure) -> list[set]:
    g = [set() for _ in structure.atoms]
    for b in structure.bonds:
        g[b.atom1.idx].add(b.atom2.idx)
        g[b.atom2.idx].add(b.atom1.idx)
    return g


def _near(graph, a, depth=3) -> set:
    """atoms within `depth` bonds of a (including a)."""
    seen, front = {a}, {a}
    for _ in range(depth):
        front = {n for f in front for n in graph[f]} - seen
        seen |= front
    return seen


def _contact(structure, graph, i, j) -> float:
    ai, aj = structure.atoms[i], structure.atoms[j]
    zi, zj = ai.atomic_number, aj.atomic_number
    for h, o, zo in ((i, j, zj), (j, i, zi)):
        if structure.atoms[h].atomic_number == 1 and zo in (7, 8):
            if any(structure.atoms[n].atomic_number in (7, 8, 16) for n in graph[h]):
                return HBOND_CONTACT
    return CONTACT_SCALE * (_VDW.get(zi, 0.17) + _VDW.get(zj, 0.17))


def _env_atoms(structure, site_mappings) -> np.ndarray:
    in_block = set()
    for m in site_mappings:
        for atoms in m.blocks.values():
            in_block |= set(atoms)
    return np.array([i for i in range(len(structure.atoms)) if i not in in_block], dtype=int)


def _block_contacts(structure, graph, x, atoms, env, tree) -> list[Contact]:
    """clashing contacts of `atoms` with env (KD-tree over env coords) and within the block."""
    atoms = list(atoms)
    out = []
    for a in atoms:
        skip = _near(graph, a)
        for k in tree.query_ball_point(x[a], _SEARCH_RADIUS):
            c = int(env[k])
            if c in skip:
                continue
            d = float(np.linalg.norm(x[c] - x[a]))
            t = _contact(structure, graph, a, c)
            if d < t:
                out.append(Contact(a, c, d, t))
    for p in range(len(atoms)):
        skip = _near(graph, atoms[p])
        for q in range(p + 1, len(atoms)):
            a, c = atoms[p], atoms[q]
            if c in skip:
                continue
            d = float(np.linalg.norm(x[c] - x[a]))
            t = _contact(structure, graph, a, c)
            if d < t:
                out.append(Contact(a, c, d, t))
    out.sort(key=lambda c: c.distance / c.contact)
    return out


def _score(contacts) -> float:
    return float(sum(c.overlap ** 2 for c in contacts))


def block_clashes(structure, site_mappings) -> list[BlockClashReport]:
    """Clashing contacts of every block, against the environment (non-block atoms) and inside
    the block. Same-site blocks never interact and are not checked against each other; blocks of
    other sites are skipped too (their pairs are lambda-product scaled)."""
    x = _coords_nm(structure)
    graph = _graph(structure)
    env = _env_atoms(structure, site_mappings)
    tree = cKDTree(x[env])
    out = []
    for s, m in enumerate(site_mappings, start=1):
        for name, atoms in m.blocks.items():
            cl = _block_contacts(structure, graph, x, atoms, env, tree)
            out.append(BlockClashReport(site=s, block=name, clashes=cl,
                                        score_before=_score(cl), score_after=_score(cl)))
    return out


def _rotate(x, idx, p0, p1, deg):
    """rotate x[idx] by `deg` about the axis p0 -> p1 (Rodrigues)."""
    if not idx or deg == 0.0:
        return
    k = (p1 - p0) / np.linalg.norm(p1 - p0)
    t = math.radians(deg)
    v = x[idx] - p1
    x[idx] = (p1 + v * math.cos(t) + np.cross(k, v) * math.sin(t)
              + np.outer(v @ k, k) * (1 - math.cos(t)))


def _subtree(graph, start, block, stop) -> list:
    """block atoms reachable from `start` without passing through `stop`; None if it loops back."""
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for n in graph[a]:
            if n == stop and a != start:
                return None                      # ring through the axis: not rotatable
            if n in block and n not in seen and n != stop:
                seen.add(n); stack.append(n)
    return sorted(seen - {start})


def _torsion_axes(structure, graph, block: set, all_blocks: set):
    """[(p0, p1, moving atoms)] for the block's chi1 (whole block about env P -> attachment A) and
    for each first-shell bond A -> X inside the block (X's subtree). Empty if the block hangs off
    more than one env atom or the attachment has no unique heavy env neighbor (e.g. proline).
    `all_blocks` = every block atom of every site: the attachment's other neighbors that belong to
    a sibling block (the competing side chains on the same atom) are not environment."""
    attach = {n for a in block for n in graph[a] if n not in block}
    if len(attach) != 1 or attach & all_blocks:
        return []
    A = next(iter(attach))
    heavy_env = [n for n in graph[A] if n not in all_blocks and structure.atoms[n].atomic_number > 1]
    if len(heavy_env) != 1:
        return []
    axes = [(heavy_env[0], A, sorted(block))]
    for X in sorted(n for n in graph[A] if n in block):
        if structure.atoms[X].atomic_number == 1:
            continue
        sub = _subtree(graph, X, block, A)
        if not sub:
            continue
        if len(sub) == 3 and all(structure.atoms[i].atomic_number == 1 for i in sub):
            continue                              # methyl / NH3+: 120-degree symmetric
        axes.append((A, X, sub))
    return axes


def relieve_clashes(structure, site_mappings, *, rotate_reference: bool = False,
                    max_axes: int = 4) -> list[BlockClashReport]:
    """For every block that clashes, try the staggered rotamers (0/120/240 degrees) of its chi1
    and first-shell torsions and keep the lowest clash score; writes the new coordinates into
    `structure`. The reference (first) block of a site is the environment's own side chain and is
    left in place unless `rotate_reference`. Returns one report per block (after = remaining)."""
    x = _coords_nm(structure)
    graph = _graph(structure)
    env = _env_atoms(structure, site_mappings)
    tree = cKDTree(x[env])
    all_blocks = set(range(len(structure.atoms))) - set(env.tolist())
    reports = []
    for s, m in enumerate(site_mappings, start=1):
        for k, (name, atoms) in enumerate(m.blocks.items()):
            block = set(atoms)
            cl = _block_contacts(structure, graph, x, block, env, tree)
            rep = BlockClashReport(site=s, block=name, clashes=cl,
                                   score_before=_score(cl), score_after=_score(cl))
            reports.append(rep)
            if not cl or (k == 0 and not rotate_reference):
                continue
            axes = _torsion_axes(structure, graph, block, all_blocks)[:max_axes]
            if not axes:
                continue
            best = (rep.score_before, None, None)
            for combo in itertools.product(_STAGGER, repeat=len(axes)):
                if not any(combo):
                    continue
                xt = x.copy()
                for (p0, p1, mov), deg in zip(axes, combo):      # chi1 first, then inner bonds
                    _rotate(xt, mov, xt[p0].copy(), xt[p1].copy(), deg)
                ct = _block_contacts(structure, graph, xt, block, env, tree)
                sc = _score(ct)
                if sc < best[0] - 1e-12:
                    best = (sc, combo, (xt, ct))
            if best[1] is not None:
                xt, ct = best[2]
                for i in block:
                    x[i] = xt[i]
                    a = structure.atoms[i]
                    a.xx, a.xy, a.xz = (float(v) * 10.0 for v in xt[i])
                rep.clashes, rep.score_after = ct, best[0]
                rep.rotation = {(p0, p1): deg for (p0, p1, _), deg in zip(axes, best[1]) if deg}
    return reports


def format_clash(structure, report: BlockClashReport) -> str:
    w = report.worst
    if w is None:
        return f"site {report.site} block {report.block}: no clash"
    def lab(i):
        a = structure.atoms[i]; r = a.residue
        return f"{r.name}{getattr(r, 'number', '')}:{a.name}"
    return (f"site {report.site} block {report.block}: {len(report.clashes)} clash(es), worst "
            f"{lab(w.atom)}-{lab(w.partner)} {w.distance:.3f} nm (< {w.contact:.3f})")
