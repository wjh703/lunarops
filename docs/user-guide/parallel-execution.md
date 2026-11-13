# Parallel execution

LunarOps runs serially by default. Two forms of parallelism are available:
MPI processes for the observation programs and the orbit program, and OpenMP
threads inside the compiled dynamics kernel. Serial, MPI and OpenMP runs give
the same results.

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

## OpenMP

An OpenMP build of the dynamics extension (see
[Installation](../getting-started/installation.md#openmp-build)) evaluates the
point-mass interactions of large body sets in parallel threads. It helps long
orbit runs with hundreds of bodies on one node:

```bash
export OMP_NUM_THREADS=32 OMP_WAIT_POLICY=PASSIVE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m lunarops run configs/lunarops_orbit_rotation_de440_environment_10a.yml
```

The Python integration loop itself stays serial. Do not combine OpenMP
threads with `--mpi`. An OpenMP build also turns off `parallelArcs`, the
forked backward arc of a two-sided `LunarOrbitPropagation` (see
[Lunar dynamics](lunar-dynamics.md)), because the OpenMP runtime is not
fork-safe.

## Slurm examples

`jobs/` contains Slurm scripts for the reference scenarios:

| Script | Mode |
|---|---|
| `residuals_mpi.sbatch` | O−C residuals over MPI |
| `lunar_orbit.sbatch` | serial orbit propagation |
| `rotation_de430_10a.sbatch` | ten-year DE430 orbit and rotation with OpenMP; checks that the extension was built with OpenMP |
| `rotation_de430_10a_mpi.sbatch` | the same over MPI |
| `rotation_de440_environment_10a.sbatch` | ten-year DE440 environment model with OpenMP |
| `rotation_de440_environment_10a_mpi.sbatch` | the same over MPI |

The rotation scripts write into a job-specific output directory by overriding
the configuration variable `outputDir` with `--set`.
