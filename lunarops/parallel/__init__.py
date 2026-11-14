"""Parallel execution of LunarOps programs over worker processes.

The package provides two master-worker task runtimes with one interface (:mod:`~lunarops.parallel.runtime`):
local worker processes (:mod:`~lunarops.parallel.process`, ``--workers N``) and MPI ranks
(:mod:`~lunarops.parallel.mpi`, ``--mpi``). Both run the same worker task handlers
(:mod:`~lunarops.parallel.tasks`) on the observation specs sent to the workers
(:mod:`~lunarops.parallel.observation_spec`) and on lunar-dynamics epoch inputs
(:mod:`~lunarops.parallel.epoch_stream`). The ``mpi: {chunksize: N}`` option of a program sets the number of
items per worker task under either runtime.
"""
