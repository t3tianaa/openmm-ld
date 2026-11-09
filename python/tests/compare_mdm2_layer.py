"""openmm_ld's scaling of MDM2 against OpenMM-MSLD with use_plugin: energy, dU/dlambda and forces at partial lambda.

Usage: python compare_mdm2_layer.py MSLD_CHECKOUT PREPARED_DIR
"""
import json
import sys

import numpy as np
import openmm as mm

from openmm_ld import load_prepared, set_lambdas

engine, prepared_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, engine)
from msld.model import MSLDModel as OldModel  # noqa: E402

REFERENCE = mm.Platform.getPlatformByName("Reference")


def evaluate(system, x, box, lam):
    ctx = mm.Context(system, mm.VerletIntegrator(0.001), REFERENCE)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(x)
    set_lambdas(ctx, lam)
    st = ctx.getState(getEnergy=True, getForces=True, getParameterDerivatives=True)
    d = st.getEnergyParameterDerivatives()
    return (st.getPotentialEnergy()._value, np.array([d[f"lambda{b}"] for b in range(1, len(lam))]),
            np.array(st.getForces(asNumpy=True)._value))


prepared = load_prepared(prepared_dir)
base = mm.XmlSerializer.serialize(prepared.system)
model = prepared.to_model()
model.scale_bond = model.scale_angle = False
old = OldModel(model.n_atoms, fnex=model.fnex)
for bl in model.blocks[1:]:
    old.add_block(site=bl.site, atoms=bl.atoms)
old.finalize()
old.scale_bond = old.scale_angle = False
system_new = mm.XmlSerializer.deserialize(base)
model.scale_system(system_new)
system_old = mm.XmlSerializer.deserialize(base)
old.scale_system(system_old, use_plugin=True)
print("forces:", system_new.getNumForces(), "new,", system_old.getNumForces(), "old")

# geometry: a cheap relaxation of the prepared positions, so forces are not dominated by clashes
ctx = mm.Context(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_new)), mm.VerletIntegrator(0.001), REFERENCE)
ctx.setPeriodicBoxVectors(*prepared.box)
ctx.setPositions(prepared.positions)
set_lambdas(ctx, [1.0] + [0.2] * 5)
mm.LocalEnergyMinimizer.minimize(ctx, 10.0, 100)
x = ctx.getState(getPositions=True).getPositions(asNumpy=True)._value

blocks = {"WT": 1, "ILE": 2, "LEU": 3, "SER": 4, "VAL": 5}


def lam(**on):
    rest = (1.0 - sum(on.values())) / (5 - len(on))
    v = [1.0] + [rest] * 5
    for name, value in on.items():
        v[blocks[name]] = value
    return v


for label, l in (("uniform", lam()), ("LEU 0.7", lam(LEU=0.7)), ("LEU 0.4 VAL 0.4", lam(LEU=0.4, VAL=0.4)), ("LEU 0.99", lam(LEU=0.99))):
    e0, d0, f0 = evaluate(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_new)), x, prepared.box, l)
    e1, d1, f1 = evaluate(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system_old)), x, prepared.box, l)
    print(json.dumps({"point": label, "E": e1, "E rel err": abs(e0 - e1) / abs(e1), "dU/dlambda max rel err":
                      float(np.max(np.abs(d0 - d1) / np.maximum(np.abs(d1), 1.0))), "dU/dlambda max abs err": float(np.max(np.abs(d0 - d1))),
                      "F max abs err": float(np.max(np.abs(f0 - f1))), "F max": float(np.max(np.abs(f1)))}), flush=True)
