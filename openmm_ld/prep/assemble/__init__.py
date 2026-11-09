"""Assemble — turn a mapping + variant structures into a hybrid OpenMM System.

  align.py     Kabsch superposition (CoordinateAligner / BackboneAligner / IdentityAligner)
  structure.py HybridStructureBuilder — align + graft blocks + transplant terms + rename
  charges.py   ChargeRenormalization (env average + block renorm to integer net + guard)
  solvate.py   water-box overlay + neutralizing ions
  system.py    HybridSystemBuilder — hybrid Structure -> plain OpenMM System (NBFIX-safe)
  hub.py       HybridSystemAssembler — runs the substeps, returns a HybridSystemBundle
  clash.py     clash check of grafted blocks + staggered-rotamer relief
  merge.py     bonded-term helpers: transplant (used by structure.py) + cross-variant guard
"""
from .align import CoordinateAligner, BackboneAligner, IdentityAligner, RigidTransform, kabsch
from .charges import (ChargeResult, EnvChargeReport, RenormReport, assert_iso_charge,
                      average_env_charges, renormalize_charges, variant_net_charges,
                      ChargeRenormalization, SpreadRenormalizer, ProportionalRenormalizer,
                      BufferAwareRenormalizer)
from .merge import (find_cross_variant_terms, assert_no_cross_variant_terms,
                    delete_cross_variant_terms)
from .solvate import solvate, SolvateResult
from .clash import block_clashes, relieve_clashes, BlockClashReport
from .structure import HybridStructureBuilder, SiteSpec, HybridResult
from .system import HybridSystemBuilder, is_solvated, strip_solvent
from .hub import HybridSystemAssembler, HybridSystemBundle

__all__ = ["CoordinateAligner", "BackboneAligner", "IdentityAligner", "RigidTransform", "kabsch",
           "ChargeResult", "EnvChargeReport", "RenormReport", "assert_iso_charge",
           "average_env_charges", "renormalize_charges", "variant_net_charges",
           "ChargeRenormalization", "SpreadRenormalizer", "ProportionalRenormalizer",
           "BufferAwareRenormalizer",
           "find_cross_variant_terms", "assert_no_cross_variant_terms",
           "delete_cross_variant_terms", "solvate", "SolvateResult",
           "block_clashes", "relieve_clashes", "BlockClashReport",
           "HybridStructureBuilder", "SiteSpec", "HybridResult",
           "HybridSystemBuilder", "is_solvated", "strip_solvent",
           "HybridSystemAssembler", "HybridSystemBundle"]
