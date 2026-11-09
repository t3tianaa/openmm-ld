"""Rigid superposition of variants onto the reference frame (Kabsch/SVD).

Phase D grafts each variant's BLOCK atoms onto the reference's shared core. The core exists
ONCE (taken from the reference), so each variant's block must first be rotated/translated into
the reference frame. The `CoordinateAligner` computes that rigid transform by least-squares
superposition of the shared-core atoms; Phase D then applies it to the block atoms.

    reference core  ●───●───●         variant (own frame)   ◦───◦───◦
                        │                                       │
                    block stays put                         block rides along
                                                                │
         align(reference, variant) ─► RigidTransform T  s.t.  T(variant core) ≈ reference core
         then T applied to the variant's BLOCK atoms places them next to the shared core.

Fit atoms: by default the rigid backbone (N, CA, C, O) — a stable frame that side-chain
flexibility cannot perturb. `MCSAligner` (ligands, later) will fit on the MCS core instead.

Duck-typed: reads only `atom.name` and `atom.xx/.xy/.xz`, so it works on ParmEd structures or
any equivalent. NumPy only (already a core dependency); no new packages.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# Rigid backbone atoms to superpose on (see module docstring).
BACKBONE_FIT = ("N", "CA", "C", "O")


@dataclass
class RigidTransform:
    """A proper rigid motion x -> R x + t (R a rotation, det R = +1)."""
    R: np.ndarray                       # (3, 3)
    t: np.ndarray                       # (3,)

    def apply(self, xyz):
        """Transform coordinates. Accepts a single point (3,) or an array (N, 3)."""
        return np.asarray(xyz, float) @ self.R.T + self.t

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix (JSON-friendly via .tolist())."""
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.t
        return M

    @classmethod
    def from_matrix(cls, M) -> "RigidTransform":
        M = np.asarray(M, float)
        return cls(M[:3, :3].copy(), M[:3, 3].copy())

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    def rmsd(self, mobile, target) -> float:
        """RMSD between T(mobile) and target (both (N, 3))."""
        diff = self.apply(mobile) - np.asarray(target, float)
        return float(np.sqrt((diff * diff).sum(axis=1).mean()))


def kabsch(mobile, target) -> RigidTransform:
    """Optimal rigid transform mapping `mobile` onto `target` (least-squares, no reflection).

    mobile, target : (N, 3) arrays of CORRESPONDING points (row i <-> row i). Needs N >= 3
    non-collinear points. Returns a RigidTransform T with T.apply(mobile) ~= target.
    """
    mobile = np.asarray(mobile, float)
    target = np.asarray(target, float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3:
        raise ValueError("mobile and target must be matching (N, 3) arrays")
    if len(mobile) < 3:
        raise ValueError(f"need >= 3 points for a rigid superposition, got {len(mobile)}")

    pc = mobile.mean(axis=0)
    qc = target.mean(axis=0)
    P = mobile - pc
    Q = target - qc
    H = P.T @ Q                                  # 3x3 cross-covariance
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))       # +/-1: fix an improper (reflection) solution
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = qc - R @ pc
    return RigidTransform(R, t)


# ---------------------------------------------------------------------
# Aligner classes
# ---------------------------------------------------------------------
class CoordinateAligner(ABC):
    """Base: return the rigid transform that maps `variant` into `reference`'s frame."""

    @abstractmethod
    def align(self, reference, variant) -> RigidTransform:
        raise NotImplementedError


class IdentityAligner(CoordinateAligner):
    """No-op aligner for inputs that are already co-framed (returns the identity)."""

    def align(self, reference=None, variant=None) -> RigidTransform:
        return RigidTransform.identity()


@dataclass
class BackboneAligner(CoordinateAligner):
    """Superpose on named fit atoms (default: the backbone N, CA, C, O).

    Matches fit atoms across `reference` and `variant` BY NAME, then runs Kabsch. The names
    must exist in both (they do for standard residues; for a backbone mutation the atoms still
    share names, only their types differ). Raises if fewer than 3 shared fit atoms are found.
    """
    fit_atoms: tuple = BACKBONE_FIT

    def align(self, reference, variant) -> RigidTransform:
        ref_xyz = {a.name: (a.xx, a.xy, a.xz) for a in reference.atoms}
        var_xyz = {a.name: (a.xx, a.xy, a.xz) for a in variant.atoms}
        names = [n for n in self.fit_atoms if n in ref_xyz and n in var_xyz]
        if len(names) < 3:
            raise ValueError(
                f"BackboneAligner needs >= 3 shared fit atoms {self.fit_atoms}; found "
                f"{names} in common")
        target = np.array([ref_xyz[n] for n in names])
        mobile = np.array([var_xyz[n] for n in names])
        return kabsch(mobile, target)
