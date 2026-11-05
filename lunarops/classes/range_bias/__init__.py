"""Additive station range-bias tables.

The package provides ``RangeBiasTable``, built from configuration rows, a YAML file or the built-in
INPOP21a table, and its ``RangeBiasComponent`` rows. Corrections are in centimetres of two-way light
distance over UTC date intervals.
"""

from .table import (
    BUILTIN_ADDITIVE_RANGE_BIAS_TABLES,
    INPOP21A_RANGE_BIAS_COMPONENTS,
    INPOP21A_RANGE_BIAS_TABLE,
    RangeBiasComponent,
    RangeBiasTable,
    builtin_additive_range_bias_table,
    load_additive_range_bias_table,
)

__all__ = [
    "BUILTIN_ADDITIVE_RANGE_BIAS_TABLES",
    "INPOP21A_RANGE_BIAS_COMPONENTS",
    "INPOP21A_RANGE_BIAS_TABLE",
    "RangeBiasComponent",
    "RangeBiasTable",
    "builtin_additive_range_bias_table",
    "load_additive_range_bias_table",
]
