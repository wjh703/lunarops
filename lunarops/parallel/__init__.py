"""MPI backend for distributed LLR forward-model tasks.

Serial execution is intentionally single-process.  Multi-rank execution is
provided by :mod:`lunarops.parallel.mpi`; launch with
``mpirun/srun python -m lunarops run cfg.yml --mpi`` and configure task size
per program with ``mpi: {chunksize: 8}``. Supported by ``LlrResiduals`` and
``LlrProcessing``.
"""
