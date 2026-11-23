#!/usr/bin/env bash
# Run both PMI networks, both legs each, one after another on one GPU:
#
#     ./run_networks.sh                       # all four legs, one production replica each (~2.2 h)
#     ./run_networks.sh val_free val_complex   # only these legs
#     REPLICAS=3 ./run_networks.sh             # three replicas instead of one
#     NONBONDED=plugin ./run_networks.sh       # skip check-gpu and force a route
#     STAGE2_CYCLES=70 ./run_networks.sh exp_free   # more ALF for a network that has not flattened
#
# Each leg is prepare -> route check -> ALF stage 1 -> ALF stage 2 -> production replicas, and every
# stage is skipped if its marker is already in out_<leg>/, so re-running the script resumes.
#
# Replicas can be added later for the cost of the extra production alone. Running with REPLICAS=1 and
# then again with REPLICAS=3 keeps the prepared system and both ALF stages and runs only replicas 2 and
# 3: about 40 min per free leg and 73 min per complex leg for each one. That is why the skip test below
# counts the replicas actually present instead of looking for one "this leg is finished" marker.
#
# The validation network runs first: its dG is what the measured Kd values test.
# Order matters for the GPU, not for correctness; the legs are independent.
set -euo pipefail
cd "$(dirname "$0")"

ALL_LEGS=(val_free val_complex exp_free exp_complex)
LEGS=("${ALL_LEGS[@]}")
if [ "$#" -gt 0 ]; then LEGS=("$@"); fi
REPLICAS=${REPLICAS:-1}

# Output directory of replica r: run, run2, run3, ...
replica_dir() {
    if [ "$1" -gt 1 ]; then echo "run$1"; else echo run; fi
}

# How many of the requested replicas already have a dG.json.
replicas_present() {
    local wd=$1 n=0 r
    for r in $(seq 1 "$REPLICAS"); do
        if [ -f "$wd/$(replica_dir "$r")/dG.json" ]; then n=$((n + 1)); fi
    done
    echo "$n"
}

# Both legs of a network have a first replica -> the binding free energies. Called after each leg and
# again at the end, so it reports each network once however the legs were ordered or skipped.
REPORTED=""
finish_network() {
    local net=$1
    local c="out_${net}_complex/run/dG.json" f="out_${net}_free/run/dG.json"
    case " $REPORTED " in *" $net "*) return 0 ;; esac
    if [ -f "$c" ] && [ -f "$f" ]; then
        REPORTED="$REPORTED $net"
        echo "[$net] ddG_bind (first replica of each leg):"
        openmm-ld ddg "$c" "$f" "ddg_${net}.json"
        if [ -f analyze_networks.py ]; then
            python analyze_networks.py "$net" 2>&1 | tee "report_${net}.txt"
        fi
    fi
}

for LEG in "${LEGS[@]}"; do
    CFG="${LEG}.toml"
    WD="out_${LEG}"
    if [ ! -f "$CFG" ]; then
        echo "no such leg: $LEG (expected $CFG); legs are ${ALL_LEGS[*]}" >&2
        exit 1
    fi
    mkdir -p "$WD"
    HAVE=$(replicas_present "$WD")
    if [ "$HAVE" -ge "$REPLICAS" ]; then
        echo "[$LEG] already has $HAVE of $REPLICAS replicas; raise REPLICAS for more"
        case "$LEG" in
            val_*) finish_network val ;;
            exp_*) finish_network exp ;;
        esac
        continue
    fi
    echo "=== $LEG ($HAVE of $REPLICAS replicas done)"

    if [ ! -f "$WD/prepared/hybrid_system.xml" ]; then
        openmm-ld prepare "$CFG" --workdir "$WD" 2>&1 | tee "$WD/prepare.log"
    fi

    # Nonbonded route, chosen once per leg and kept in nonbonded.choice:
    #   plugin = the MSLD GPU kernel (fast; needs libOpenMMMSLDCUDA built, see examples/mdm2_gpu/README.md)
    #   stock  = stock OpenMM forces with a finite-difference dU/dlambda (slower, no build needed)
    # check-gpu writes its log and the log is then grepped. Never pipe check-gpu into grep -q: grep exits
    # at the first match, check-gpu dies of SIGPIPE while printing the timings, and under pipefail a
    # passing check then looks like a failed one.
    if [ -n "${NONBONDED:-}" ]; then
        echo "$NONBONDED" > "$WD/nonbonded.choice"
    elif [ ! -f "$WD/nonbonded.choice" ]; then
        rm -f "$WD/check-gpu.log"
        if python -c "import msldplugin" 2>/dev/null; then
            openmm-ld check-gpu "$CFG" --workdir "$WD" > "$WD/check-gpu.log" 2>&1 || true
            cat "$WD/check-gpu.log"
        fi
        if grep -q "kernel AGREES" "$WD/check-gpu.log" 2>/dev/null; then
            echo plugin > "$WD/nonbonded.choice"
        else
            echo stock > "$WD/nonbonded.choice"
        fi
    fi
    NB=$(cat "$WD/nonbonded.choice")
    echo "[$LEG] nonbonded route: $NB"

    # ALF stage 1: short cycles from zero biases (cycles/steps from the config).
    if [ ! -f "$WD/.stage1" ]; then
        openmm-ld alf "$CFG" --workdir "$WD" --set system.nonbonded="$NB" 2>&1 | tee -a "$WD/alf.log"
        touch "$WD/.stage1"
    fi
    # ALF stage 2: ALF's standard 100 ps cycles (25 ps discarded), continuing from stage 1's biases.
    # alf.cycles is a cumulative target, so this carries on from the config's 30 to STAGE2_CYCLES.
    if [ ! -f "$WD/.stage2" ]; then
        openmm-ld alf "$CFG" --workdir "$WD" --set system.nonbonded="$NB" \
            --set alf.cycles="${STAGE2_CYCLES:-50}" --set alf.steps=25000 --set alf.equil_steps=6250 \
            2>&1 | tee -a "$WD/alf.log"
        touch "$WD/.stage2"
    fi

    # Production: independent replicas from the same flattened state, differing only in the random seed.
    # One replica is enough for a first number; several give an error bar from their spread, and stop a
    # state missing from one run from making the bootstrap undefined.
    for r in $(seq 1 "$REPLICAS"); do
        OUT=$(replica_dir "$r")
        if [ -f "$WD/$OUT/dG.json" ]; then
            echo "[$LEG] replica $r already done"
            continue
        fi
        echo "[$LEG] production replica $r of $REPLICAS -> $WD/$OUT"
        openmm-ld run "$CFG" --workdir "$WD" --set system.nonbonded="$NB" \
            --set run.output="$OUT" --set dynamics.seed="$r" 2>&1 | tee -a "$WD/run.log"
    done

    case "$LEG" in
        val_*) finish_network val ;;
        exp_*) finish_network exp ;;
    esac
done

for NET in val exp; do
    finish_network "$NET"
done
