#!/usr/bin/env bash
# Run one leg on the GPU: prepare -> ALF stage 1 (short cycles) -> ALF stage 2 (100 ps cycles) -> production.
# Usage:  ./run_gpu.sh complex   |   ./run_gpu.sh free        (both can share one GPU)
# Every stage resumes where it stopped if you run the script again.
set -euo pipefail
LEG=${1:?complex or free}
CFG=${LEG}.toml
WD=out_${LEG}
mkdir -p "$WD"
cp -n *.pdb "$WD"/ 2>/dev/null || true
cp "$CFG" "$WD"/
cd "$WD"
[ -f prepared/hybrid_system.xml ] || openmm-ld prepare "$CFG" 2>&1 | tee prepare.log
# Nonbonded route, chosen once per leg and kept in nonbonded.choice:
#   plugin = the MSLD GPU kernel (fast; needs libOpenMMMSLDCUDA built, see README),
#   stock  = stock OpenMM forces with a finite-difference dU/dlambda (slower, no build needed).
# A leg that already started ALF without a choice file stays on stock.  Override with NONBONDED=plugin|stock.
if [ -n "${NONBONDED:-}" ]; then
    echo "$NONBONDED" > nonbonded.choice
elif [ ! -f nonbonded.choice ]; then
    if [ -d alf ]; then
        echo stock > nonbonded.choice
    else
        # Log first, then grep the file: "| grep -q" exits at the first match, which kills tee and
        # check-gpu with SIGPIPE, and under pipefail the whole test then fails (stock despite AGREES).
        rm -f check-gpu.log
        if python -c "import msldplugin" 2>/dev/null; then
            openmm-ld check-gpu "$CFG" > check-gpu.log 2>&1 || true
            cat check-gpu.log
        fi
        if grep -q "kernel AGREES" check-gpu.log 2>/dev/null; then
            echo plugin > nonbonded.choice
        else
            echo stock > nonbonded.choice
        fi
    fi
fi
NB=$(cat nonbonded.choice)
echo "nonbonded route: $NB"
# stage 1: 30 cycles x (10 + 40 ps)
openmm-ld alf "$CFG" --set system.nonbonded=$NB 2>&1 | tee -a alf.log
# stage 2: up to cycle 50 with ALF's standard 100 ps cycles (25 ps discarded)
openmm-ld alf "$CFG" --set system.nonbonded=$NB --set alf.cycles=${STAGE2_CYCLES:-50} --set alf.steps=25000 --set alf.equil_steps=6250 2>&1 | tee -a alf.log
# production (2 ns by default; ALF's estimator, bootstrap over 5 chunks)
openmm-ld run "$CFG" --set system.nonbonded=$NB 2>&1 | tee run.log
