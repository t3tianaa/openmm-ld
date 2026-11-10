"""The openmm-ld command on the peptide alone (dry, fast), against the library calls it stands for.

Runs compared by random stream are in separate processes, and read the energy at the same points, since each read
advances the integrator's random stream (docs/msld_integrator.md).
"""
import filecmp
import json
import os
import shutil
import subprocess
import sys

import openmm as mm
import pytest

import compare_prep
from openmm_ld import LambdaDynamics, load_prepared
from openmm_ld.alf import add_flattening_biases, flatten
from openmm_ld.cli import _build, load_config, main

CONFIG = """
[prepare]
forcefield = ["amber14-all.xml"]
solvate = false
[[prepare.environment]]
file = "mdm2_peptide.pdb"
add_hydrogens = true
[[prepare.site]]
residue = 3
variants = { ILE = "ILE_var.pdb", LEU = "LEU_var.pdb", SER = "SER_var.pdb", VAL = "VAL_var.pdb" }
[system]
scale_bond = false
scale_angle = false
[dynamics]
platform = "Reference"
seed = 3
[minimize]
max_iterations = 5
[alf]
iterations = 2
steps_per_iter = 50
sample_every = 10
max_step = 8.0
tol = 0.0
[run]
steps = 40
metrics_every = 10
trajectory_every = 20
"""


@pytest.fixture
def workspace(tmp_path):
    for name in os.listdir(compare_prep.MDM2):
        if name.endswith(".pdb"):
            shutil.copy(os.path.join(compare_prep.MDM2, name), tmp_path)
    config = tmp_path / "toy.toml"
    config.write_text(CONFIG)
    return config


def test_prepare_matches_the_library(workspace):
    import openmm_ld.prep as prep
    assert main(["prepare", str(workspace)]) == 0
    bundle = compare_prep.build_peptide(prep, str(workspace.parent))
    direct = workspace.parent / "direct"
    paths = bundle.save(str(direct))
    for key, path in paths.items():
        assert filecmp.cmp(path, workspace.parent / "prepared" / os.path.basename(path), shallow=False), key


def _library_run(directory):
    prepared = load_prepared(os.path.join(directory, "prepared"))
    names = [n for s in prepared.mapping.sites for n in s.block_names]
    options = {"theta_mass": 0.05, "theta_friction": 1.0}
    model = prepared.to_model(block_options={n: options for n in names})
    model.scale_bond = model.scale_angle = False
    add_flattening_biases(model)
    model.scale_system(prepared.system)
    dyn = LambdaDynamics(prepared.system, model, prepared.positions, platform="Reference", box=prepared.box, seed=3)
    dyn.potential_energy()
    dyn.minimize(max_iterations=5)
    dyn.potential_energy()
    report = flatten(dyn, n_iters=2, steps_per_iter=50, sample_every=10, max_step=8.0, tol=0.0, verbose=False)
    dyn.run(40, metrics=os.path.join(directory, "library.csv"), metrics_every=10, mapping=prepared.mapping)
    with open(os.path.join(directory, "library.json"), "w") as fh:
        json.dump({"biases": report["biases"], "couplings": report["couplings"]}, fh)


def _in_new_process(*args):
    subprocess.run([sys.executable, *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_all_reproduces_the_library_calls(workspace):
    d = str(workspace.parent)
    assert main(["prepare", str(workspace)]) == 0
    _in_new_process("-m", "openmm_ld.cli", "all", str(workspace))
    _in_new_process(os.path.abspath(__file__), d)
    with open(os.path.join(d, "alf", "biases.json")) as fh:
        saved = json.load(fh)
    with open(os.path.join(d, "library.json")) as fh:
        library = json.load(fh)
    assert saved["couplings"] == library["couplings"]
    assert saved["biases"] == library["biases"]
    assert filecmp.cmp(os.path.join(d, "library.csv"), os.path.join(d, "run", "metrics.csv"), shallow=False)


def test_run_continues_from_the_alf_state(workspace, capsys):
    d = str(workspace.parent)
    assert main(["prepare", str(workspace)]) == 0
    assert main(["alf", str(workspace)]) == 0
    assert main(["run", str(workspace), "--set", "run.steps=10", "--set", "run.metrics_every=10"]) == 0
    with open(os.path.join(d, "alf", "state.xml")) as fh:
        state = mm.XmlSerializer.deserialize(fh.read())
    parameters = state.getParameters()
    with open(os.path.join(d, "alf", "biases.json")) as fh:
        saved = json.load(fh)
    assert parameters["biascoef1"] == pytest.approx(saved["biases"]["1"])
    # theta comes back with the state: lambda from the restored theta equals lambda saved in the Context
    _, _, dyn = _build(load_config(workspace), d, saved)
    dyn.context.setState(state)
    assert dyn.get_lambdas()[1] == pytest.approx(parameters["lambda1"], abs=1e-12)
    assert dyn.get_lambdas()[1] != pytest.approx(0.2, abs=1e-3)
    assert os.path.exists(os.path.join(d, "run", "metrics.csv"))


def test_config_errors_are_clear(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[nonsense]\nx = 1\n")
    with pytest.raises(SystemExit, match="unknown section"):
        load_config(bad)
    bad.write_text("[run]\nstep = 1\n")
    with pytest.raises(SystemExit, match="unknown key"):
        load_config(bad)
    bad.write_text("[run]\nsteps = 1\n")
    assert load_config(bad, ["run.steps=7"])["run"]["steps"] == 7


if __name__ == "__main__":
    import msldplugin
    if os.environ.get("MSLD_REFERENCE_PLUGIN"):
        msldplugin.loadReferencePlugin(os.environ["MSLD_REFERENCE_PLUGIN"])
    _library_run(sys.argv[1])
