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
# stage 1: 30 cycles x (10 + 40 ps)
openmm-ld alf "$CFG" 2>&1 | tee -a alf.log
# stage 2: up to cycle 50 with ALF's standard 100 ps cycles (25 ps discarded)
openmm-ld alf "$CFG" --set alf.cycles=${STAGE2_CYCLES:-50} --set alf.steps=25000 --set alf.equil_steps=6250 2>&1 | tee -a alf.log
# production (2 ns by default; ALF's estimator, bootstrap over 5 chunks)
openmm-ld run "$CFG" 2>&1 | tee run.log
