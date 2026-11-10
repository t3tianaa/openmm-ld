"""Mapping — discover which atoms are shared environment vs each variant block.

  core.py      CoreMapping · HybridBlueprint · Mapper/ManualMapper · build_hybrid_local_mapping
               · model_from_mappings   (the data model; ParmEd-free)
  backbone.py  AminoAcidBackboneMapper (peptides: shared core by atom NAME + TYPE)
  mcs.py       MCSMapper (ligands: shared core by graph MCS; RDKit) · molecular_graph
"""
from .core import (CoreMapping, HybridBlueprint, Mapper, ManualMapper,
                   model_from_mappings, build_hybrid_local_mapping)
from .backbone import AminoAcidBackboneMapper
from .mcs import MCSMapper, molecular_graph

__all__ = ["CoreMapping", "HybridBlueprint", "Mapper", "ManualMapper",
           "model_from_mappings", "build_hybrid_local_mapping",
           "AminoAcidBackboneMapper", "MCSMapper", "molecular_graph"]
