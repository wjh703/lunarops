"""Typed YAML reports and orbit metadata."""

from __future__ import annotations

from pathlib import Path

from .yaml_artifact import read_structured_text, write_structured_text


def write_processing_report(path, payload) -> Path:
    return write_structured_text(path, "processingReport", payload)


def read_processing_report(path):
    return read_structured_text(path, "processingReport")


def write_lunar_orbit_metadata(path, payload) -> Path:
    return write_structured_text(path, "lunarOrbitMetadata", payload)


def read_lunar_orbit_metadata(path):
    return read_structured_text(path, "lunarOrbitMetadata")
