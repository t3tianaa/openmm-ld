#!/usr/bin/env bash
# Run both PMI networks, both legs each, one after another on one GPU:
#
#     ./run_networks.sh                      # all four legs, in order
#     ./run_networks.sh val_free val_complex  # only these legs
#     REPLICAS=1 ./run_networks.sh            # one production replica instead of three
#     NONBONDED=plugin ./run_networks.sh      # skip check-gpu and force a route
#     STAGE2_CYCLES=90 ./run_networks.sh exp_free   # more ALF for a network that has not flattened
#
# Each leg is prepare -> route check -> ALF stage 1 -> ALF stage 2 -> production replicas, and every
# stage is skipped if its marker file is already in out_<leg>/, so re-running the script resumes.
# The validation network runs first: its dG is what the measured Kd values test.
# Order matters for the GPU, not for correctness; the legs are independent.
set -euo pipefail
cd "$(dirname "$0")"

ALL_LEGS=(val_free val_complex exp_free exp_complex)
LEGS=("${ALL_LEGS[@]}")
if [ "$#" -gt 0 ]; then LEGS=("$@"); fi
REPLICAS=${REPLICAS:-3}

# Both legs of a network done -> the binding free energies.
finish_network() {
    local net=$1
    local c="out_${net}_complex/run/dG.json" f="out_${net}_free/run/dG.json"
    if [ -f "$c" ] && [ -f "$f" ]; then
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
    if [ -f "$WD/.done" ]; then
        echo "[$LEG] already finished ($WD/.done); delete that file to run it again"
        continue
    fi
    mkdir -p "$WD"
    echo "=== $LEG"

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
    if [ ! -f "$WD/.stage2" ]; then
        openmm-ld alf "$CFG" --workdir "$WD" --set system.nonbonded="$NB" \
            --set alf.cycles="${STAGE2_CYCLES:-60}" --set alf.steps=25000 --set alf.equil_steps=6250 \
            2>&1 | tee -a "$WD/alf.log"
        touch "$WD/.stage2"
    fi

    # Production: independent replicas from the same flattened state, differing only in the random seed.
    # Several short replicas beat one long run here: the error bar comes from their spread, and a state
    # that is never visited in one replica no longer makes the bootstrap undefined.
    for r in $(seq 1 "$REPLICAS"); do
        OUT=run
        if [ "$r" -gt 1 ]; then OUT="run${r}"; fi
        if [ -f "$WD/$OUT/dG.json" ]; then
            echo "[$LEG] replica $r already done"
            continue
        fi
        echo "[$LEG] production replica $r of $REPLICAS -> $WD/$OUT"
        openmm-ld run "$CFG" --workdir "$WD" --set system.nonbonded="$NB" \
            --set run.output="$OUT" --set dynamics.seed="$r" 2>&1 | tee -a "$WD/run.log"
    done

    touch "$WD/.done"
    case "$LEG" in
        val_*) finish_network val ;;
        exp_*) finish_network exp ;;
    esac
done

for NET in val exp; do
    finish_network "$NET"
done
