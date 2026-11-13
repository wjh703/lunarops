# Parallel execution

LunarOps runs serially by default. Parallelism comes from MPI worker
processes for the observation programs and the orbit program; the compiled
extensions are single-threaded. Serial and MPI runs give the same results.

## MPI

Launch the run under an MPI launcher with `--mpi`:

```bash
mpirun -n 16 python -m lunarops run config.yml --mpi
srun python -m lunarops run config.yml --mpi
```

`--mpi` needs `mpi4py` (the `mpi` extra) and at least two ranks: rank 0
executes the program sequence and ranks 1 to N−1 work for it. With a single
rank the run fails; run without `--mpi` instead.

What the workers do depends on the program:

| Program | Work sent to workers |
|---|---|
| `LlrResiduals` | blocks of normal points; workers return result rows |
| `LlrProcessing` | blocks of normal points at every linearization; workers return equation batches |
| `LlrObservationPrediction` | blocks of grid epochs |
| `LunarOrbitPropagation` | time-dependent epoch inputs (ephemeris states, Earth axes, interactions between prescribed bodies); rank 0 integrates |

`mpi.chunksize` (default 8) sets the number of normal points or epochs per
task. Each worker builds its own model from a broadcast description: it maps
the ephemeris kernels itself, while the parsed EOP table is sent from rank 0. Rank 0
restores the original order before writing, and tasks have deterministic
identities, so the output does not depend on the number of ranks or on the
order in which tasks finish. All other programs run on rank 0.

### Threads per rank

Use one native thread per MPI rank. `--mpi` sets `OPENBLAS_NUM_THREADS`,
`OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before NumPy is loaded. Pass the
same variables to remote ranks when the launcher does not forward the
environment (`mpirun -x OMP_NUM_THREADS ...`). Rank 0 solves the normal
equations of `LlrProcessing` itself; they are small, and the single-thread
policy applies there too.

## Slurm examples

`jobs/` contains Slurm scripts for the reference scenarios:

| Script | Mode |
|---|---|
| `residuals_mpi.sbatch` | O−C residuals over MPI |
| `lunar_orbit.sbatch` | serial orbit propagation |
| `rotation_de430_10a.sbatch` | ten-year DE430 orbit and rotation, serial |
| `rotation_de430_10a_mpi.sbatch` | the same over MPI |
| `rotation_de440_environment_10a.sbatch` | ten-year DE440 environment model, serial |
| `rotation_de440_environment_10a_mpi.sbatch` | the same over MPI |

The rotation scripts write into a job-specific output directory by overriding
the configuration variable `outputDir` with `--set`.
