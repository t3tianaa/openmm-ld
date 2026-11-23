# Two MSLD networks on the PMI / MDM2 system

Both networks sit on MDM2(25-109) and a 12-mer of the PMI family, the phage-display peptide of
[Pazgier et al., PNAS 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856455). Each network is run as two
legs, `complex` (MDM2 + peptide) and `free` (peptide alone), and
ddG_bind = dG_complex - dG_free. Negative means the state binds MDM2 more tightly than the reference.

| | network V (validation) | network E (exploration) |
|---|---|---|
| background peptide | `mdm2_peptide.pdb`, TSFAEYWALLSP (PMI-N8A, [3LNZ](https://www.rcsb.org/structure/3LNZ)) | `pmi_L10A_peptide.pdb`, TSFAEYWNLASP (PMI-L10A) |
| sites | position 8, position 10 | positions 4, 9, 11 |
| blocks per site | 4, 2 | 4, 2, 2 |
| states | 8 | 16 |
| with measured Kd | 6 | 4 |
| reference for ddG | PMI (`ASN+WT`) | PMI-L10A (`WT+WT+WT`) |
| configs | `val_free.toml`, `val_complex.toml` | `exp_free.toml`, `exp_complex.toml` |

## Network V: does MSLD reproduce measured affinities?

Two sites, every state a sequence someone has measured or a near neighbour of one:

* **position 8** — WT is ALA here, because the background peptide is PMI-N8A. Blocks ASN (restores PMI),
  LEU, THR.
* **position 10** — WT is LEU. Block ALA, the L10A mutation that costs 3.4 kcal/mol.

| state | peptide | sequence | Kd (nM) | study | ddG vs PMI |
|---|---|---|---|---|---|
| `ASN+WT` | PMI | TSFAEYWNLLSP | 2.7 ± 0.5 | L2021 | 0 (reference) |
| `WT+WT` | PMI-N8A | TSFAEYWALLSP | 0.49 ± 0.21 | L2010 | −1.01 |
| `WT+ALA` | PMI-L10A+N8A | TSFAEYWALASP | 229 ± 11 | L2021 | +2.63 |
| `LEU+ALA` | PMI-L10A+N8L | TSFAEYWLLASP | 344 ± 8 | L2021 | +2.87 |
| `ASN+ALA` | PMI-L10A | TSFAEYWNLASP | 897 ± 33 | L2021 | +3.44 |
| `THR+ALA` | PMI-L10A+N8T | TSFAEYWTLASP | 962 ± 29 | L2021 | +3.48 |
| `LEU+WT` | — | TSFAEYWLLLSP | — | — | prediction |
| `THR+WT` | — | TSFAEYWTLLSP | — | — | prediction |

Six measured states spanning **4.5 kcal/mol** (0.49 nM to 962 nM, a factor of 2000). That span is the
point: the four N8 variants on the L10A background sit within 0.85 kcal/mol of each other, which is too
close to rank with this much sampling, but L10A against PMI is a 3.4 kcal/mol signal that any working
implementation has to get. So read the result as one large test plus four fine ones.

The 2010 and 2021 studies both measured PMI-L10A, at 890 and 897 nM. They agree, which is what makes it
defensible to use the 2010 value for PMI-N8A alongside 2021 values for everything else. `analyze_networks.py`
marks that row with `*` anyway.

## Network E: three sites, 16 states, 12 of them never made

Everything sits on the PMI-L10A background, because that is the background the 2021 single mutations were
measured on.

* **position 4** — WT ALA. Blocks LEU (**A4L, measured**, 98 nM: the largest single gain in the series),
  ILE, VAL.
* **position 9** — WT LEU. Block GLN (**L9Q, measured**, 535 nM).
* **position 11** — WT SER. Block MET (**S11M, measured**, 198 nM).

| state | peptide | Kd (nM) | ddG vs PMI-L10A |
|---|---|---|---|
| `WT+WT+WT` | PMI-L10A | 897 ± 33 | 0 (reference) |
| `LEU+WT+WT` | PMI-L10A+A4L | 98 ± 5 | −1.31 |
| `WT+WT+MET` | PMI-L10A+S11M | 198 ± 4 | −0.90 |
| `WT+GLN+WT` | PMI-L10A+L9Q | 535 ± 21 | −0.31 |
| the other 12 | — | — | predictions |

Two questions it asks:

1. **Is the A4L gain about volume or about shape?** ILE is Leu's branched isomer at the same volume, VAL is
   one CH2 smaller. If the gain is buried surface, VAL < LEU ≈ ILE. If it is shape, ILE and LEU separate.
2. **Do the favourable mutations add up?** They were measured one at a time. Experiment says they do not:
   against the sum of its own singles, PMI-M4 is **+1.07** kcal/mol and PMI-M5 **+0.76** kcal/mol short
   (anti-cooperative). The corner `LEU+GLN+MET` tests whether MSLD reproduces a positive non-additivity of
   that size, and it is a new candidate — the combination was never synthesized. `analyze_networks.py`
   prints this comparison.

### Why PMI-M4 and PMI-M5 are not in the network

Both were on the list and neither can be reached by a side-chain site:

* **PMI-M4** (LTFLEYWNLASP) needs T1L. Position 1 is the peptide N-terminus, and prep refuses it:
  `env atom N charge varies by 0.597 e across variants (> 0.5); it is not really a shared atom.` The
  N-terminal residue's backbone carries the NH3+ charges, the grafted internal residue does not, so N is
  not a shared atom. `mapper = "backbone"` fails the same way. This is a correct refusal, not a bug.
* **PMI-M5** (TSFAEYWAQAMQ) needs P12Q. Proline's ring closes through its backbone N, so Pro→X is a
  backbone mutation by construction, and position 12 is the C-terminus as well.

Network E covers three of M5's five changes (L9Q, S11M and L10A in the background) and M4's strongest
single (A4L). M5 itself is two changes away from `LEU+GLN+MET`, M4 is two away.

To reach M4 or M5 exactly, the peptide would have to be capped — acetyl on the N-terminus, amide on the
C-terminus — so that positions 1 and 12 become internal residues. That is worth checking against the
papers anyway: if the measured peptides were capped and this model gives them charged termini, then the
T1, S2 and P12 comparisons carry a systematic error, and every network here that touches positions 4-11 is
unaffected. Capping is a change to prep, not to a config.

PMI-M2 and PMI-M3 are left out deliberately: 8 and 7 changes from PMI, far outside one network, and M3 at
3.7 pM makes optimizing this system moot.

## Running it

```bash
cd examples/pmi_networks
./run_networks.sh                     # all four legs, validation network first
```

Each leg is prepare -> check-gpu -> ALF stage 1 -> ALF stage 2 -> production replicas, writing to
`out_<leg>/`. Every stage leaves a marker, so re-running resumes rather than repeating. Named legs
(`./run_networks.sh val_free val_complex`) run only those. `REPLICAS=1` and `STAGE2_CYCLES=90` are the
other knobs; `NONBONDED=plugin` skips the kernel check. Run outputs land in `out_<leg>/`, which
`.gitignore` excludes along with the `ddg_*.json` and `report_*` files, so nothing from a run can be
committed.

Production is **three 2 ns replicas** from the same flattened state, differing only in the random seed. That is
the fix for the error bars in the first MDM2 run: a bootstrap over chunks of one trajectory treats
correlated chunks as independent and gave ±0.24 kcal/mol where the chunk-to-chunk spread was 1.9. The
replicas are fed to the published ALF estimator as the independent runs it expects, and a state missing
from one replica no longer makes the error undefined.

Rough cost, scaling from the 0.93 ms/step the free leg reached on the CUDA VM. The complex leg is 26,300
atoms against 5,700, so 3.5 ms/step is a guess, not a measurement:

| leg | steps | estimate |
|---|---|---|
| `val_free` | 2.7 M | 0.7 h |
| `val_complex` | 2.7 M | 2.6 h |
| `exp_free` | 2.7 M | 0.7 h |
| `exp_complex` | 2.7 M | 2.6 h |

About 6.6 h for all four, and the validation network runs first, so its ddG lands after roughly 3.3 h.
Each leg is ALF stage 1 (40 cycles x 12.5k steps), stage 2 (20 more at 31.25k) and three 2 ns replicas.
`REPLICAS=2` takes about 1.5 h off the total and costs you a third of the error-bar statistics;
`./run_networks.sh val_free val_complex` runs the validation network alone in 3.3 h.

## Reading the result

```bash
python analyze_networks.py val      # or exp, or both
```

It pools the replicas, prints ddG_bind against the measured value per state with MAE, RMSE and rank
correlation, and writes `report_<net>.json`. `run_networks.sh` runs it automatically once both legs of a
network are finished.

What to check before believing any of it:

* **`alf.log`**: `scaling` approaching 1, `frac(l>0.99)` non-zero for *every* block, `b` stopped drifting,
  T_theta near 300 K. Network E has 8 blocks; if some never light up, raise `STAGE2_CYCLES`.
* **endpoint counts** in the report: 16 states share the sampling, so each gets a fraction of what 5 states
  got. A state with a few hundred endpoint samples is noise, as LEU was in the first MDM2 run.
* **the measured spread**: network E's four anchors span only 1.3 kcal/mol. A calculation that cannot beat
  1.3 kcal/mol of error says nothing about ranking them.

## Files

| | |
|---|---|
| `make_inputs.py` | builds `ALA/ASN/GLN/MET/THR_var.pdb` from complete residues of `../mdm2_gpu/mdm2_protein.pdb`, and `pmi_L10A_peptide.pdb` from `../mdm2_gpu/mdm2_peptide.pdb` by A8N + L10A with PDBFixer |
| `*_var.pdb` | side-chain donors; `ILE/LEU/VAL_var.pdb` are copies from `examples/mdm2_gpu` |
| MDM2 and PMI-N8A structures | not copied here: the configs read `../mdm2_gpu/mdm2_protein.pdb` and `../mdm2_gpu/mdm2_peptide.pdb` |
| `run_networks.sh` | all four legs, resumable |
| `analyze_networks.py` | replicas pooled, compared with the measured Kd table |

Measured values are same-study SPR on MDM2(25-109) in 10 mM HEPES / 150 mM NaCl, pH 7.4:
**L2021** = [Li et al., Acta Pharm Sin B 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8463443),
doi:10.1016/j.apsb.2021.06.010, PMI reference 2.7 nM;
**L2010** = [Li et al., J Mol Biol 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856455),
doi:10.1016/j.jmb.2010.03.005, PMI reference 3.2 nM. Both report Kd at 25 °C (2010 says room temperature);
the ddG columns use RT at 298.15 K.
