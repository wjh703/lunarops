"""Parallel execution of LunarOps programs over MPI ranks.

The package provides the master-worker MPI runtime, the observation specs sent to the workers and the
distributed computation of lunar-dynamics epoch data. A run uses it when started with ``--mpi``; the
``mpi: {chunksize: N}`` option of a program sets the number of items per worker task.
"""
