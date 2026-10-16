# lunarops

GROOPS-inspired Lunar Laser Ranging processing system.

Project documentation is maintained in [docs/README.md](docs/README.md).

For Linux OpenMP source-gravity batches, build with
`LUNAROPS_ENABLE_OPENMP=1 uv pip install --no-build-isolation --editable .`.
Run with `OMP_NUM_THREADS=32 OMP_WAIT_POLICY=PASSIVE`, keeping
`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The DE430/DE440 ten-year Slurm
scripts request 32 CPUs and check that the extension has OpenMP enabled.
The default build is serial; both builds use the same pair summation order.
