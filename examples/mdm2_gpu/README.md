# MDM2 peptide site, GPU run (stock-force route)

Two legs, same site and variants: `complex` (MDM2 + peptide) and `free` (peptide alone).
ΔΔG_bind(WT→X) = ΔG_complex(WT→X) − ΔG_free(WT→X).

```bash
conda activate msld                     # OpenMM 8.6.1 with CUDA (conda-forge), numpy, scipy, parmed, pdbfixer
pip install --no-deps -e ../..          # the openmm-ld Python layer (the C++ plugin is not needed for this route)
./run_gpu.sh free    > free.out 2>&1 &
./run_gpu.sh complex > complex.out 2>&1 &
# progress: grep '\[ALF\]' out_*/alf.log
# after production of both legs:
openmm-ld ddg out_complex/run/dG.json out_free/run/dG.json ddg.json
```

What to watch in `alf.log`: `scaling` should approach 1, `frac(l>0.99)` should become non-zero for every
variant, and `b` should stop drifting. T_theta should stay near 300 K.
