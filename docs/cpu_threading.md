# The nonbonded force on several CPU threads

The CPU platform's `Threads` property now applies to `MSLDNonbondedForce`. It's the same kernel as on Reference (double
precision, no SIMD), with two changes in `platforms/reference/src/ReferenceMSLDKernels.cpp`:

- **Threads over the pairs.** The direct-space pair list is split in a fixed way over OpenMM's `ThreadPool`. Each thread has its
  own energy, force and dU/dλ accumulators, added up in thread order at the end.
- **A reused neighbour list.** Measuring showed this mattered more than the threads: building the list took 0.22 s of the 0.38 s
  evaluation, and it can't be split over threads with OpenMM's builder. The list now reaches 0.1 nm past the cutoff and is rebuilt only
  when an atom has moved more than half of that, or the box has changed. The padding shrinks if half the box requires it.

PME stays serial: at 0.014 s per evaluation it doesn't dominate.

## Timings

MDM2, 26,305 atoms, CPU platform on a 10-core Mac (4 performance, 6 efficiency), a step of `MSLDIntegrator` with the plugin force,
from a relaxed start at 300 K; seconds. The neighbour list was built 13 times in 121 evaluations, each build about 0.29 s.

| Threads | Step | Direct pairs | Neighbour list (averaged) | PME (reciprocal + charge derivatives) | Rest (integrator, constraints, bonded) |
|---|---|---|---|---|---|
| 1 | 0.257 | 0.200 | 0.031 | 0.014 | 0.012 |
| 2 | 0.158 | 0.102 | 0.031 | 0.014 | 0.011 |
| 4 | 0.113 | 0.057 | 0.031 | 0.014 | 0.011 |
| 10 | 0.097 | 0.041 | 0.030 | 0.014 | 0.011 |

Before this work a step took about 0.40 s. Stock OpenMM's `NonbondedForce` alone on the same positions, on the same CPU platform with
single-precision SIMD: 0.042, 0.023, 0.013 and 0.0098 s per evaluation at 1, 2, 4 and 10 threads. It computes no dU/dλ, no
soft-core and no block scaling, and it isn't a dynamics step, since unscaled the competing blocks overlap. So the plugin's force is still
about 6 to 9 times slower than stock nonbonded (the step minus "Rest"), with most of that in the double-precision pair loop.

Beyond 4 threads the gain is small: the efficiency cores help little, and the list and PME are serial. At 10 threads 1 ns takes about 27 hours.

## Results don't depend on the threads

`python/tests/measure_cpu_threads.py` on MDM2, against one thread, at the prepared positions:

| Threads | Energy, relative | dU/dλ, max absolute (kJ/mol) | Force, max absolute (kJ/mol/nm) | Repeat gives identical bytes |
|---|---|---|---|---|
| 2 | 2.2e-13 | 1.1e-13 | 2.8e-9 | |
| 4 | 2.5e-13 | 3.4e-13 | 2.8e-9 | |
| 10 | 3.5e-13 | 2.3e-13 | 2.8e-9 | Yes |

The largest force at those positions has clashes in it, hence the force figure. On a small system `TestReferenceMSLDNonbondedForce` checks 1, 2, 4 and 7
threads with no cutoff, both reaction-field methods and PME, through small moves and a jump in one Context, against a new Reference
Context at each checked position: energy within 1e-11, forces and dU/dλ within 1e-10. It also checks that a box change rebuilds the list.

The result can depend slightly, at rounding level, on the thread count and on when the list was last rebuilt, because that changes the
order of the sums. It doesn't vary between runs of the same sequence.

## What would help next

In order of the time left at 4 threads: the pair loop (single precision and SIMD, or a GPU kernel), then a larger padding or a threaded
list builder, then PME.
