# Two MSLD networks on the PMI / MDM2 system

Both networks sit on MDM2(25-109) and a 12-mer of the PMI family, the phage-display peptide of
[Pazgier et al., PNAS 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856455). Each network is run as two
legs, `complex` (MDM2 + peptide) and `free` (peptide alone), and
ddG_bind = dG_complex - dG_free. Negative means the state binds MDM2 more tightly than the reference.

| | network V (validation) | network E (exploration) |
|---|---|---|
| background peptide | both: `../mdm2_gpu/mdm2_peptide.pdb`, TSFAEYWALLSP (PMI-N8A, [3LNZ](https://www.rcsb.org/structure/3LNZ)) | |
| sites | position 8, position 10 | positions 8, 9, 11 |
| blocks per site | 4, 2 | 2, 3, 2 |
| states | 8 | 12 |
| with measured Kd | 6 | 2 |
| what it is for | testing the calculation against affinities spanning 4.5 kcal/mol | designing toward PMI-M3's 8-11 motif |
| reference for ddG | PMI (`ASN+WT`) | PMI (`ASN+WT+WT`) |
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

## Network E: three sites, twelve states, the PMI-M3 motif

This one is **design, not validation** — network V carries the validation load. It sits on the same
PMI-N8A peptide as network V, `../mdm2_gpu/mdm2_peptide.pdb`, and opens three sites with substitutions
from the PMI-M3/M5 series:

* **position 8** — WT ALA. Block ASN, which restores wild-type PMI.
* **position 9** — WT LEU. Blocks GLN and MET.
* **position 11** — WT SER. Block MET.

2 x 3 x 2 = 12 states. The point of the network is the corner **`WT+GLN+MET`**, sequence
**TSFAEYWAQLMP**: Ala8, Gln9, Leu10, Met11 is exactly PMI-M3's 8-11 motif. M3 itself is LTFLEYWAQLMQ at
**3.7 pM**, the tightest MDM2 peptide in the series, so that motif is known to be worth having. The
network asks how much of it survives on the plain PMI backbone, without M3's T1L, S2T, A4L and P12Q.

| state | peptide | sequence | Kd (nM) | study |
|---|---|---|---|---|
| `ASN+WT+WT` | PMI | TSFAEYWNLLSP | 2.7 ± 0.5 | L2021, reference |
| `WT+WT+WT` | PMI-N8A | TSFAEYWALLSP | 0.49 ± 0.21 | L2010 |
| `WT+GLN+MET` | — | TSFAEYWAQLMP | — | **the target: M3's 8-11 motif** |
| the other 9 | — | — | — | predictions |

**Only two states have a measured Kd on this background, and that is the network's real weakness.** The
increments it is built from were measured on the PMI-L10A background, not this one: N8A −0.81, L9Q −0.31,
L9M −0.29, S11M −0.90 kcal/mol against PMI-L10A at 897 nM. Add the three that make up the corner and
naive additivity predicts roughly −2 kcal/mol from PMI, which is a few hundred pM. Treat that as the
number to beat, not a prediction: the same arithmetic fails by **+0.76 kcal/mol** for PMI-M5 and
**+1.07** for PMI-M4 against their own singles, both anti-cooperative. `analyze_networks.py` prints the
MSLD version of that sum against the corner, so you can see whether the calculation reproduces
anti-cooperativity of that size.

What the network can and cannot tell you:

* It **can** rank the twelve, say whether the M3 motif is worth having without M3's other four changes,
  and say whether MSLD thinks the increments add up.
* It **cannot** be checked against experiment except at its two anchors, and those two differ by only
  1.0 kcal/mol. A sign error or a 2 kcal/mol systematic would pass unnoticed here, which is why network V
  runs first.

### Why PMI-M3, M4 and M5 are not themselves states

M3, M4 and M5 all need changes outside what a side-chain site can do:

* **T1L** (in M2, M3, M4). Position 1 is the peptide N-terminus, whose backbone carries the NH3+ charges
  that an internal residue does not, so the shared N is not actually shared. Prep refuses it:
  `env atom N charge varies by 0.597 e across variants (> 0.5); it is not really a shared atom.`
  `mapper = "backbone"` fails the same way. This is a correct refusal, not a bug.
* **P12Q** (in M2, M3, M5). Proline's ring closes through its own backbone nitrogen, so Pro to anything
  is a backbone mutation by construction, and position 12 is the C-terminus as well.

So the closest reachable thing is the motif, not the molecule. To reach M3, M4 or M5 exactly, the peptide
would have to be capped, acetyl on the N-terminus and amide on the C-terminus, so positions 1 and 12
become internal residues. That is worth checking against the papers in any case: if the measured peptides
were capped and this model gives them charged termini, every comparison at positions 1, 2 and 12 carries a
systematic error. Nothing between positions 4 and 11 is affected, so both networks here are clear of it.

A4L is also left out, even though it is the largest single gain in the 2021 series at 98 nM, because it
would make this a four-site network. It is the obvious next one.

## Running it

```bash
cd examples/pmi_networks
./run_networks.sh                     # all four legs, validation network first
```

Each leg is prepare -> check-gpu -> ALF stage 1 -> ALF stage 2 -> production replicas, writing to
`out_<leg>/`. Every stage leaves a marker, so re-running resumes rather than repeating. Named legs
(`./run_networks.sh val_free val_complex`) run only those. `REPLICAS=3` and `STAGE2_CYCLES=70` are the
other knobs; `NONBONDED=plugin` skips the kernel check. Run outputs land in `out_<leg>/`, which
`.gitignore` excludes along with the `ddg_*.json` and `report_*` files, so nothing from a run can be
committed.

Production is **three 2 ns replicas** from the same flattened state, differing only in the random seed. That is
the fix for the error bars in the first MDM2 run: a bootstrap over chunks of one trajectory treats
correlated chunks as independent and gave ±0.24 kcal/mol where the chunk-to-chunk spread was 1.9. They are
fed to the published ALF estimator as the independent runs it expects, and a state missing from one replica
no longer makes the error undefined. Adding them later costs only the extra production; see below.

Cost, from the step rates your own VM logs give: **0.92 ms/step** for a free leg (375,000 steps in 348 s
and 625,000 in 573 s) and **1.67 ms/step** for a complex leg (375,000 in 633 s and 625,000 in 1032 s).
No estimates in this table. Each leg is 25,000 steps of equilibration, ALF stage 1 (30 x 12,500) and
stage 2 (20 x 31,250), then 525,000 steps per 2 ns replica:

| | free leg | complex leg | one network | both networks |
|---|---|---|---|---|
| `REPLICAS=1`, the default | 24 min | 43 min | **1.1 h** | **2.2 h** |
| each further replica | +8 min | +15 min | +23 min | +46 min |
| `REPLICAS=3` | 40 min | 73 min | 1.9 h | 3.7 h |

The default is one replica, which is exactly the single-site run you have already done: 30 ALF cycles, 20
more, one 2 ns production with `chunks = 5`. So the default plan is that run four times over, about 2.2 h.

**Replicas can be added later for the price of the extra production alone.** Run with `REPLICAS=1`, look
at the numbers, and then run the same command with `REPLICAS=3`: the prepared system and both ALF stages
are kept, and only replicas 2 and 3 are run. Nothing is recomputed and nothing is wasted. ALF is the
expensive half of a leg, so going from one replica to three costs 46 minutes across all four legs, not
another full run.

`STAGE2_CYCLES=40` halves ALF's long cycles and takes about 40 min off the total, at the risk of biases
that have not settled.

### What one 2 ns replica buys, and what the third one adds

Sampling, not wall clock, decides whether these numbers mean anything. At `lambda_every = 10` each replica
records 50,000 lambda samples. The single-site MDM2 run spent 12% of its samples at some endpoint; taking
that as the rate here:

| | states | endpoint samples per state, on average |
|---|---|---|
| the earlier single-site run, 2 ns, 1 replica | 5 | about 1,200 |
| network V, 1 replica / 3 replicas | 8 | about 750 / 2,200 |
| network E, 1 replica / 3 replicas | 12 | about 500 / 1,500 |

So at one replica **both networks are more thinly sampled per state than the run you have already seen**,
because the same sampling is divided among 8 or 12 states instead of 5. That average also hides a lot: in
the single-site run the per-state counts ran from 172 to 3,511, a factor of 20, because ALF never flattens
the populations perfectly. Expect some states in the low hundreds, and at least one that cannot be
estimated at all.

With one replica the error bar is also back to a bootstrap over the five chunks of a single trajectory,
which is what reported +/-0.24 kcal/mol on LEU where the chunk-to-chunk spread was 1.9, and NaN where a
chunk had no endpoint samples. **Read a one-replica run as a ranking and a sanity check, not as a number
with an error bar.** The third replica is what turns it into one, and it is 46 minutes away at any time.

## Reading the result

```bash
python analyze_networks.py val      # or exp, or both
```

It pools the replicas, prints ddG_bind against the measured value per state with MAE, RMSE and rank
correlation, and writes `report_<net>.json`. `run_networks.sh` runs it automatically once both legs of a
network are finished.

What to check before believing any of it:

* **`alf.log`**: `scaling` approaching 1, `frac(l>0.99)` non-zero for *every* block, `b` stopped drifting,
  T_theta near 300 K. Network E has 7 blocks; if some never light up, raise `STAGE2_CYCLES`.
* **endpoint counts** in the report: 12 states share the sampling, so each gets a fraction of what 5 states
  got. A state with a few hundred endpoint samples is noise, as LEU was in the first MDM2 run.
* **the measured spread**: network E has only two anchors and they differ by 1.0 kcal/mol, so it cannot
  catch a systematic error. That is network V's job, and V should be read first.

## Files

| | |
|---|---|
| `make_inputs.py` | builds the five new `*_var.pdb` donors from complete residues of `../mdm2_gpu/mdm2_protein.pdb` |
| `*_var.pdb` | the six side-chain donors; `LEU_var.pdb` is a copy from `examples/mdm2_gpu` |
| MDM2 and PMI-N8A structures | not copied here: both networks read `../mdm2_gpu/mdm2_protein.pdb` and `../mdm2_gpu/mdm2_peptide.pdb` |
| `run_networks.sh` | all four legs, resumable |
| `analyze_networks.py` | replicas pooled, compared with the measured Kd table |

Measured values are same-study SPR on MDM2(25-109) in 10 mM HEPES / 150 mM NaCl, pH 7.4:
**L2021** = [Li et al., Acta Pharm Sin B 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8463443),
doi:10.1016/j.apsb.2021.06.010, PMI reference 2.7 nM;
**L2010** = [Li et al., J Mol Biol 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856455),
doi:10.1016/j.jmb.2010.03.005, PMI reference 3.2 nM. Both report Kd at 25 °C (2010 says room temperature);
the ddG columns use RT at 298.15 K.
