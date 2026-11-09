"""A prepared system on disk: what prep (stage P2) will write and what a run starts from.

  <prefix>_system.xml    the plain, not yet lambda-scaled OpenMM System
  <prefix>_mapping.json  which atoms form each block, per site
  <prefix>_start.pdb     the topology, in the System's atom order
  <prefix>_positions.npy positions in nm, in full precision
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import openmm as mm
from openmm import app, unit

from .model import model_from_mapping


class _Site:
    """One site of a saved mapping: what the output writers read."""

    def __init__(self, blocks):
        self.blocks = {name: set(atoms) for name, atoms in blocks.items()}

    @property
    def block_names(self):
        return list(self.blocks)

    def all_block_atoms(self):
        return set().union(*self.blocks.values()) if self.blocks else set()


class Mapping:
    def __init__(self, data):
        self.data = data
        self.n_atoms = int(data["n_atoms"])
        self.sites = [_Site(s["blocks"]) for s in data["sites"]]


@dataclass
class PreparedSystem:
    system: mm.System
    mapping: Mapping
    topology: app.Topology
    positions: unit.Quantity
    box: tuple

    def to_model(self, **kw):
        """The finalized MSLDModel for this system."""
        return model_from_mapping(self.mapping.data, **kw)


def load_prepared(directory, prefix="hybrid") -> PreparedSystem:
    path = lambda suffix: os.path.join(directory, f"{prefix}_{suffix}")
    with open(path("system.xml")) as fh:
        system = mm.XmlSerializer.deserialize(fh.read())
    with open(path("mapping.json")) as fh:
        mapping = Mapping(json.load(fh))
    pdb = app.PDBFile(path("start.pdb"))
    if system.getNumParticles() != mapping.n_atoms or pdb.topology.getNumAtoms() != mapping.n_atoms:
        raise ValueError(f"{directory}: system, mapping and start PDB disagree on the number of atoms")
    positions = np.load(path("positions.npy")) * unit.nanometer
    return PreparedSystem(system, mapping, pdb.topology, positions, system.getDefaultPeriodicBoxVectors())
