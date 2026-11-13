# Parallel execution

LunarOps runs serially by default. Parallelism comes from worker processes
that evaluate the models for the main process, either local processes on one
machine (`--workers N`) or MPI ranks (`--mpi`); the compiled extensions are
single-threaded. Serial, worker-process and MPI runs give the same results.

| | `--workers N` | `--mpi` |
|---|---|---|
| Machines | one | one or more |
| Launch | `python -m lunarops run config.yml --workers N` | under `mpirun`, `mpiexec` or `srun` |
| Requires | nothing beyond the base install | `mpi4py` (the `mpi` extra) and an MPI library |
| Processes | the main process and N workers | N ranks: rank 0 and N−1 workers |
| Slurm | one task with `--cpus-per-task` | one task per rank (`--ntasks`) |

Use `--workers` on a single machine or a single Slurm node: it needs no MPI
installation and no launcher. Use `--mpi` when the work should spread over
several nodes, for example the O−C residuals of a large data set. The two
options cannot be combined.

## Local worker processes

```bash
python -m lunarops run config.yml --workers 8
```

`--workers N` (N ≥ 1) starts N worker processes on this machine before the
configuration is read. The main process executes the program sequence and
the workers evaluate model tasks for it. The workers are started afresh
(the `spawn` start method) and ignore Ctrl-C: an interrupt stops the main
process, which then terminates the workers at once. A worker that dies ends
the run with an error naming the worker, its process ID and its exit code.
If the main process itself is killed, the workers notice their closed pipe
and exit.

Give each process a CPU: N workers keep N + 1 CPUs busy. A two-sided orbit
run with `parallelArcs` (the default) keeps N + 2 busy: its backward arc runs
in a process of its own, which takes about B/(F+B) of the N workers for itself
(B and F the backward and forward durations; see
[Lunar dynamics](lunar-dynamics.md#integration)), so the CPU budget is
N + 2 whatever the split. A serial two-sided run keeps two CPUs busy. The run
logs a warning when fewer CPUs are usable than processes are busy.

## MPI

Launch the run under an MPI launcher with `--mpi`:

```bash
mpirun -n 16 python -m lunarops run config.yml --mpi
srun python -m lunarops run config.yml --mpi
```

`--mpi` needs `mpi4py` (the `mpi` extra) and at least two ranks: rank 0
executes the program sequence and ranks 1 to N−1 work for it. With a single
rank the run fails; run without `--mpi` instead. An interrupt aborts the whole
MPI job.

## What the workers do

The work sent to the workers depends on the program, and is the same for
both kinds of workers:

| Program | Work sent to workers |
|---|---|
| `LlrResiduals` | blocks of normal points; workers return result rows |
| `LlrProcessing` | blocks of normal points at every linearization; workers return equation batches |
| `LlrObservationPrediction` | blocks of grid epochs |
| `LunarOrbitPropagation` | time-dependent epoch inputs (ephemeris states, Earth axes, interactions between prescribed bodies); the main process integrates |

`mpi.chunksize` (default 8) sets the number of normal points or epochs per
task, with either kind of workers. Each worker builds its own model from a
broadcast description: it maps the ephemeris kernels itself, while the parsed
EOP table is sent from the main process. The main process restores the
original order before writing, and tasks have deterministic identities, so
the output does not depend on the number of workers or on the order in which
tasks finish. All other programs run in the main process.

### Threads per process

Use one native thread per process. `--workers` and `--mpi` set
`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` to 1 before
NumPy is loaded; worker processes inherit them. Under MPI, pass the same
variables to remote ranks when the launcher does not forward the environment
(`mpirun -x OMP_NUM_THREADS ...`). The main process solves the normal
equations of `LlrProcessing` itself; they are small, and the single-thread
policy applies there too.

## Slurm examples

`jobs/` contains Slurm scripts for the reference scenarios:

| Script | Mode |
|---|---|
| `residuals_mpi.sbatch` | O−C residuals over MPI, across nodes |
| `lunar_orbit.sbatch` | orbit propagation with worker processes |
| `rotation_de430_10a.sbatch` | ten-year DE430 orbit and rotation, serial |
| `rotation_de430_10a_workers.sbatch` | the same with worker processes |
| `rotation_de440_environment_10a.sbatch` | ten-year DE440 environment model, serial |
| `rotation_de440_environment_10a_workers.sbatch` | the same with worker processes |

A worker-process job is one Slurm task with several CPUs:

```bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=9
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
workers=$(( SLURM_CPUS_PER_TASK - 1 ))   # the main process; - 2 for two-sided orbit runs
.venv/bin/python -u -m lunarops run CONFIG --workers "$workers" --working-dir .
```

The rotation scripts write into a job-specific output directory by overriding
the configuration variable `outputDir` with `--set`.
