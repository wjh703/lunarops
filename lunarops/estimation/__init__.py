"""Nonlinear least-squares estimation for LLR.

Three layers, each depending only on the ones above it:

``lsq``
    Least-squares machinery without LLR or stochastic-model concepts: dense linearization, absolute normal equations
    ``N x = W`` and their strict Cholesky solution, row blocks, block elimination of random effects,
    pseudo-observations, and the parameter and covariance products.
``stochastic``
    The stochastic model of the LLR observations: variance components, the variance function and its estimation
    (LS-VCE), night sessions, robust weights and their consistency factors, and outlier diagnostics.
``earth_orientation``, ``constraints``
    Random Earth-orientation corrections per station night and parameter constraints, built on ``lsq``.
``adjustment``
    The LLR adjustment: plan and processing steps, observation screening, the iterative ``LlrAdjustmentSolver``,
    result models, convergence test and reporting.
"""
