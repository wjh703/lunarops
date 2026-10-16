"""Strict readers and writers for the current report and metadata contracts."""

from __future__ import annotations

from pathlib import Path

from lunarops.base.serialization import plain_data
from lunarops.classes.time import Epoch, TimeScale

from .report_schema import LUNAR_ORBIT_METADATA_SCHEMA, PROCESSING_REPORT_SCHEMA
from .yaml_artifact import read_structured_text, write_structured_text

REPORT_VERSION = 2
METADATA_VERSION = 2


def _nonempty(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")


def _positive(value, path, *, zero_allowed=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or (value < 0 if zero_allowed else value <= 0):
        raise ValueError(f"{path} must be {'non-negative' if zero_allowed else 'positive'}.")


def validate_processing_report(payload):
    result = PROCESSING_REPORT_SCHEMA.validate(plain_data(payload), path="processingReport")
    reason = result["terminationReason"]
    _nonempty(reason, "processingReport.terminationReason")
    converged = result["converged"]
    if converged != (reason == "CONVERGED"):
        raise ValueError("processingReport.converged contradicts terminationReason.")
    summary = result["summary"]
    for field, expected in (("converged", converged), ("terminationReason", reason)):
        if field in summary and summary[field] != expected:
            raise ValueError(f"processingReport.summary.{field} contradicts report status.")
    quality = result["publicationQuality"]
    eligible = quality["formalProductEligible"]
    expected_status = "passed" if eligible else "diagnostic"
    if quality["status"] != expected_status or bool(quality["reasons"]) == eligible:
        raise ValueError("processingReport.publicationQuality status/reasons contradict formalProductEligible.")
    for index, item in enumerate(quality["reasons"]):
        _nonempty(item, f"processingReport.publicationQuality.reasons[{index}]")
    if eligible and not converged:
        raise ValueError("processingReport.publicationQuality cannot approve a nonconverged result.")
    if not converged and reason not in quality["reasons"]:
        raise ValueError("processingReport.publicationQuality.reasons must include terminationReason.")
    parameter_names = []
    for index, parameter in enumerate(result["parameters"]):
        path = f"processingReport.parameters[{index}]"
        for field in ("name", "unit"):
            _nonempty(parameter[field], f"{path}.{field}")
        parameter_names.append(parameter["name"])
        for field in ("cofactorUncertaintyM", "formalUncertaintyM"):
            if parameter.get(field) is not None:
                _positive(parameter[field], f"{path}.{field}", zero_allowed=True)
        if eligible:
            for field in ("formalUncertaintyM", "finalValue"):
                if parameter.get(field) is None:
                    raise ValueError(f"{path}.{field} is required for a formal product.")
    if len(set(parameter_names)) != len(parameter_names):
        raise ValueError("processingReport.parameters has duplicate names.")
    if "parameterNames" in result and result["parameterNames"] != parameter_names:
        raise ValueError("processingReport.parameterNames contradicts parameters.")
    identities = []
    for index, observation in enumerate(result["observations"]):
        path = f"processingReport.observations[{index}]"
        for field in ("observationId", "stationId", "epochUtc"):
            _nonempty(observation[field], f"{path}.{field}")
        Epoch.from_isot(observation["epochUtc"], scale=TimeScale.UTC)
        identities.append(observation["observationId"])
        _positive(observation["aprioriSigmaM"], f"{path}.aprioriSigmaM")
        for field in ("sigma0M", "sigmaFactor", "baseVarianceM2", "baseWeightPerM2"):
            if field in observation:
                _positive(observation[field], f"{path}.{field}")
        for field in ("residualSigmaM", "minimumValidSigmaM", "equivalentWeightPerM2"):
            if observation.get(field) is not None:
                _positive(observation[field], f"{path}.{field}", zero_allowed=True)
        for field in ("appliedWeightFactor", "finalStateProposedWeightFactor"):
            if field in observation and not 0 <= observation[field] <= 1:
                raise ValueError(f"{path}.{field} must be in [0, 1].")
        # Failed light-time diagnostics may omit unusable residuals, but may
        # never be marked eligible. Converged observations have numeric residuals.
        if observation["lightTimeConverged"]:
            for field in ("currentStateResidualM", "residualSigmaM"):
                if observation.get(field) is None:
                    raise ValueError(f"{path}.{field} is required when lightTimeConverged is true.")
        elif eligible:
            raise ValueError(f"{path}.lightTimeConverged disqualifies a formal product.")
        if eligible and observation.get("troposphereClamped", False):
            raise ValueError(f"{path}.troposphereClamped disqualifies a formal product.")
    if len(set(identities)) != len(identities):
        raise ValueError("processingReport.observations has duplicate observationId values.")
    for name, value in summary.items():
        if isinstance(value, int) and not isinstance(value, bool) and value < 0:
            raise ValueError(f"processingReport.summary.{name} must be non-negative.")
    if "parameterCount" in summary and summary["parameterCount"] != len(parameter_names):
        raise ValueError("processingReport.summary.parameterCount contradicts parameters.")
    if "rank" in summary and summary["rank"] > len(parameter_names):
        raise ValueError("processingReport.summary.rank exceeds parameter count.")
    if eligible:
        if "observationCount" in summary and summary["observationCount"] > len(identities):
            raise ValueError("processingReport.summary.observationCount exceeds supplied observations.")
        if summary.get("sigma0Post") is None:
            raise ValueError("processingReport.summary.sigma0Post is required for a formal product.")
        _positive(summary["sigma0Post"], "processingReport.summary.sigma0Post", zero_allowed=True)
        if summary.get("observationCount", len(identities)) <= len(parameter_names):
            raise ValueError("processingReport has insufficient posterior redundancy for a formal product.")
        if summary.get("rank") != len(parameter_names):
            raise ValueError("processingReport.summary.rank must be full for a formal product.")
        if any(item.get("updateStatus") == "INSUFFICIENT_REDUNDANCY" for item in result.get("varianceComponents", [])):
            raise ValueError("processingReport has insufficient variance-component redundancy for a formal product.")
    return result


def write_processing_report(path, payload) -> Path:
    return write_structured_text(path, "processingReport", validate_processing_report(payload), version=REPORT_VERSION)


def read_processing_report(path):
    return validate_processing_report(read_structured_text(path, "processingReport", version=REPORT_VERSION))


def validate_lunar_orbit_metadata(payload):
    return LUNAR_ORBIT_METADATA_SCHEMA.validate(plain_data(payload), path="lunarOrbitMetadata")


def write_lunar_orbit_metadata(path, payload) -> Path:
    return write_structured_text(
        path, "lunarOrbitMetadata", validate_lunar_orbit_metadata(payload), version=METADATA_VERSION
    )


def read_lunar_orbit_metadata(path):
    return validate_lunar_orbit_metadata(read_structured_text(path, "lunarOrbitMetadata", version=METADATA_VERSION))
