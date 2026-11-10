"""System builder (ported from OpenMM-MSLD msld/builder.py).

Takes an already constructed OpenMM `System` + `MSLDModel` and makes every energy term block-scaled by
per-block `lambda<b>`. There are no force groups or finite differences here: every lambda-dependent force
reports dU/dlambda analytically.
"""
from __future__ import annotations

import openmm as mm

from . import forces
from .forces import lambda_name


def lambda_parameter_names(model) -> list[str]:
    """Return names of all non-environment blocks."""
    return [lambda_name(b) for b in range(1, model.n_blocks)]


def set_lambdas(context: mm.Context, lambdas) -> None:
    """Set lambda{b} context globals from a per-block sequence (index 0 = env)."""
    present = context.getParameters()
    for b in range(1, len(lambdas)):
        name = lambda_name(b)
        if name in present:
            context.setParameter(name, float(lambdas[b]))


def set_biases(context: mm.Context, model) -> None:
    """Push the model's bias coefficients into the Context globals (the ALF update step).

    Only globals a force actually registered are set. The fixed linear bias of block b lives in
    `biascoef{b}` (from `Block.lambda_bias`); the m-th variable bias' strength in `biask{m}` (from
    `VariableBias.k`). Build the System with `model.bias_flattening = True` (or with non-zero biases)
    so those globals are present.
    """
    present = context.getParameters()
    for b in range(1, model.n_blocks):
        name = forces.bias_coef_name(b)
        if name in present:
            context.setParameter(name, float(model.blocks[b].lambda_bias))
    for m, vb in enumerate(model.variable_biases):
        name = forces.variable_bias_k_name(m)
        if name in present:
            context.setParameter(name, float(vb.k))


def bias_parameter_names(system: mm.System) -> list[str]:
    """Names of the tunable bias coefficients in a scaled System (for the integrator's watch list)."""
    names = []
    for f in system.getForces():
        if f.getName() == "MSLDBias":
            for g in range(f.getNumGlobalParameters()):
                name = f.getGlobalParameterName(g)
                if name.startswith(("biascoef", "biask")):
                    names.append(name)
    return names


def _check_unhandled(system: mm.System, model) -> None:
    """Raise if a not yet supported force type contains alchemical atoms."""
    alch = {a for a, b in enumerate(model.atom_block) if b != 0}
    if not alch:
        return
    for f in system.getForces():
        if isinstance(f, mm.CMAPTorsionForce):
            for i in range(f.getNumTorsions()):
                atoms = f.getTorsionParameters(i)[1:]      # (map, a1..a4, b1..b4)
                if any(a in alch for a in atoms):
                    raise NotImplementedError(
                        "CMAP scaling is not implemented; base System has an alchemical CMAP torsion")


def scale_msld_system(system: mm.System, model, check_unhandled: bool = True) -> dict:
    """Apply MSLD block-scaling to `system` in place.

    Returns an info dict describing what was scaled. Raises (unless check_unhandled=False) if
    unsupported force types carry alchemical atoms.
    """
    if not getattr(model, "_finalized", False):
        raise RuntimeError("call model.finalize() before scale_msld_system()")

    info: dict = {"lambda_parameters": lambda_parameter_names(model)}
    # bonded scaling can be turned off per term type (BLaDE "removescaling")
    info["scaled_bond_forces"] = (
        [f.getName() for f in forces.add_scaled_bonds(system, model)]
        if getattr(model, "scale_bond", True) else [])
    info["scaled_angle_forces"] = (
        [f.getName() for f in forces.add_scaled_angles(system, model)]
        if getattr(model, "scale_angle", True) else [])
    info["scaled_torsion_forces"] = (
        [f.getName() for f in forces.add_scaled_torsions(system, model)]
        if getattr(model, "scale_torsion", True) else [])
    info["scaled_improper_forces"] = (
        [f.getName() for f in forces.add_scaled_impropers(system, model)]
        if getattr(model, "scale_torsion", True) else [])
    force, scaled = forces.replace_nonbonded_with_plugin(system, model)
    info["nonbonded_force"] = force.getName()
    info["scaled_14_exceptions"] = scaled
    bias = forces.add_biases(system, model)
    info["bias_force"] = bias.getName() if bias is not None else None
    restr = forces.add_atom_restraints(system, model)
    info["atom_restraint_force"] = restr.getName() if restr is not None else None
    if check_unhandled:
        _check_unhandled(system, model)
    return info
