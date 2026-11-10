"""System preparation for MSLD (ported from OpenMM-MSLD msld/prep).

Turns input files (a shared "environment" + several variant "blocks") into a OpenMM System and 
an MSLDModel.
    prep   ->  HybridSystemBundle {structure, System, mapping}   (bundle.to_model() = MSLDModel)
    engine ->  scale_msld_system(system, model)  +  LambdaDynamics(...)

Stages::
    prep.ingest    files -> parametrized ParmEd Structure (DataIngestPipeline, recover_charmm_lj)
    prep.mapping   which atoms are env / each block (CoreMapping, AminoAcidBackboneMapper,
                   MCSMapper, ManualMapper, HybridBlueprint)
    prep.assemble  build the hybrid System (align, HybridStructureBuilder, ChargeRenormalization,
                   solvate, HybridSystemBuilder, HybridSystemAssembler)
"""
from .mapping import (CoreMapping, HybridBlueprint, Mapper, ManualMapper,
                      model_from_mappings, build_hybrid_local_mapping,
                      AminoAcidBackboneMapper, MCSMapper, molecular_graph)
from .ingest import (DataIngestPipeline, ingest, resolve_source, load_structure,
                     load_amber, load_charmm, apply_forcefield,
                     AmberInput, CharmmInput, ForceFieldInput, recover_charmm_lj)
from .assemble import (CoordinateAligner, BackboneAligner, IdentityAligner, RigidTransform,
                       kabsch, ChargeResult, EnvChargeReport, RenormReport, assert_iso_charge,
                       average_env_charges, renormalize_charges, variant_net_charges,
                       ChargeRenormalization, SpreadRenormalizer, ProportionalRenormalizer,
                       BufferAwareRenormalizer,
                       find_cross_variant_terms, assert_no_cross_variant_terms,
                       delete_cross_variant_terms, solvate, SolvateResult,
                       HybridStructureBuilder, SiteSpec, HybridResult, HybridSystemBuilder,
                       is_solvated, strip_solvent, HybridSystemAssembler, HybridSystemBundle)
from .build import build_msld_system, Site, Part

__all__ = [
    # mapping
    "CoreMapping", "HybridBlueprint", "Mapper", "ManualMapper",
    "model_from_mappings", "build_hybrid_local_mapping",
    "AminoAcidBackboneMapper", "MCSMapper", "molecular_graph",
    # ingest
    "DataIngestPipeline", "ingest", "resolve_source", "load_structure",
    "load_amber", "load_charmm", "apply_forcefield",
    "AmberInput", "CharmmInput", "ForceFieldInput", "recover_charmm_lj",
    # assemble
    "CoordinateAligner", "BackboneAligner", "IdentityAligner", "RigidTransform", "kabsch",
    "ChargeResult", "EnvChargeReport", "RenormReport", "assert_iso_charge",
    "average_env_charges", "renormalize_charges", "variant_net_charges",
    "ChargeRenormalization", "SpreadRenormalizer", "ProportionalRenormalizer",
    "BufferAwareRenormalizer",
    "find_cross_variant_terms", "assert_no_cross_variant_terms", "delete_cross_variant_terms",
    "solvate", "SolvateResult", "HybridStructureBuilder", "SiteSpec", "HybridResult",
    "HybridSystemBuilder", "is_solvated", "strip_solvent",
    "HybridSystemAssembler", "HybridSystemBundle",
    # facade
    "build_msld_system", "Site", "Part",
]
