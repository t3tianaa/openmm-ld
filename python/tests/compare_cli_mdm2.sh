#!/bin/sh
# openmm-ld against examples/prepare_mdm2.py and run_mdm2.py on a short MDM2 run, same seed.
# Usage: compare_cli_mdm2.sh OPENMM_LD_COMMAND OPENMM_LD_SOURCE WORKDIR   (PYTHONPATH and MSLD_REFERENCE_PLUGIN set)
set -e
cli="$1"; src="$2"; w="$3"
rm -rf "$w"; mkdir -p "$w/cli"
overrides="--set minimize.max_iterations=20 --set alf.steps_per_iter=50 --set run.steps=200"
"$cli" prepare "$src/examples/mdm2/mdm2.toml" --workdir "$w/cli" | tail -1
python "$src/examples/prepare_mdm2.py" "$w/example_prepared" | tail -1
for f in system.xml mapping.json start.pdb positions.npy topology.pdb residue.xml; do
    cmp "$w/cli/prepared/hybrid_$f" "$w/example_prepared/hybrid_$f" && echo "prepare: hybrid_$f identical"
done
MIN_ITERS=20 python "$src/examples/run_mdm2.py" "$w/example_prepared" "$w/example_run" 2 50 200 | tail -4
"$cli" all "$src/examples/mdm2/mdm2.toml" --workdir "$w/cli" $overrides | tail -3
cmp "$w/cli/run/metrics.csv" "$w/example_run/metrics.csv" && echo "all: metrics.csv identical"
# the DCD header holds a creation time, so compare the frames: everything after the header
python - "$w/cli/run/traj.dcd" "$w/example_run/traj.dcd" <<'PY'
import sys
a, b = (open(p, "rb").read() for p in sys.argv[1:])
print("all: traj.dcd", len(a), "bytes,", "identical" if a == b else
      ("frames identical" if len(a) == len(b) and a[300:] == b[300:] else "DIFFERENT"))
PY
cp -R "$w/cli/alf" "$w/alf_from_all"
"$cli" alf "$src/examples/mdm2/mdm2.toml" --workdir "$w/cli" $overrides | tail -2
cmp "$w/cli/alf/biases.json" "$w/alf_from_all/biases.json" && echo "alf: biases.json same as from all"
"$cli" run "$src/examples/mdm2/mdm2.toml" --workdir "$w/cli" $overrides | tail -3
