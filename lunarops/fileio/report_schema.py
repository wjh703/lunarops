"""Single-name contracts for current reports; validation never renames fields."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lunarops.base.serialization import plain_data


@dataclass(frozen=True)
class ReportField:
    name: str
    schema: ReportSchema
    required: bool = False
    nullable: bool = False


@dataclass(frozen=True)
class ReportSchema:
    kind: str
    fields: tuple[ReportField, ...] = ()
    item: ReportSchema | None = None
    unit: str = "1"

    def validate(self, value, *, path="report"):
        if self.kind == "configuration":
            from lunarops.config.lunar_models import LUNAR_ORBIT_SNAPSHOT_SCHEMA, validate_config_snapshot

            return validate_config_snapshot(value, LUNAR_ORBIT_SNAPSHOT_SCHEMA, path=path)
        if self.kind == "scientificResource":
            if not isinstance(value, Mapping):
                raise TypeError(f"{path} must be an object.")
            schema = KERNEL_DIRECTORY if "kernels" in value else SCIENTIFIC_FILE
            return schema.validate(value, path=path)
        if self.kind == "opaque":
            return plain_data(value)
        if self.kind == "array":
            value = plain_data(value)
            if not isinstance(value, list):
                raise TypeError(f"{path} must be an array.")
            assert self.item is not None
            return [self.item.validate(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
        if self.kind in {"object", "identifierMap", "parameterState"}:
            if not isinstance(value, Mapping):
                raise TypeError(f"{path} must be an object.")
            if self.kind == "parameterState":
                return {
                    str(key): (STATION_STATE if key == "stationRangeBias" else OPAQUE).validate(
                        item, path=f"{path}.{key}"
                    )
                    for key, item in value.items()
                }
            if self.kind == "identifierMap":
                assert self.item is not None
                return {str(key): self.item.validate(v, path=f"{path}[{key}]") for key, v in value.items()}
            lookup = {f.name: f for f in self.fields}
            unknown = set(value) - set(lookup)
            if unknown:
                raise ValueError(f"{path} has unknown field(s): {sorted(str(k) for k in unknown)}")
            result: dict[str, Any] = {}
            for f in self.fields:
                field_path = f"{path}.{f.name}"
                if f.name not in value:
                    if f.required:
                        raise ValueError(f"{field_path} is missing required field.")
                    continue
                if value[f.name] is None:
                    if not f.nullable:
                        raise ValueError(f"{field_path} must not be null.")
                    result[f.name] = None
                else:
                    result[f.name] = f.schema.validate(value[f.name], path=field_path)
            return result
        value = plain_data(value)
        expected = {"number": (int, float), "integer": (int,), "boolean": (bool,), "string": (str,)}[self.kind]
        if not isinstance(value, expected) or (self.kind in {"number", "integer"} and isinstance(value, bool)):
            raise TypeError(f"{path} must be {self.kind}.")
        return value

    def describe(self):
        result: dict[str, Any] = {"type": self.kind, "unit": self.unit}
        if self.kind == "object":
            result["fields"] = [
                {"name": f.name, "required": f.required, "nullable": f.nullable, **f.schema.describe()}
                for f in self.fields
            ]
        if self.kind in {"identifierMap", "parameterState"}:
            result["freeformIdentifiers"] = True
        if self.item:
            result["items"] = self.item.describe()
        if self.kind == "scientificResource":
            result["oneOf"] = [SCIENTIFIC_FILE.describe(), KERNEL_DIRECTORY.describe()]
        if self.kind == "configuration":
            from lunarops.config.lunar_models import LUNAR_ORBIT_SNAPSHOT_SCHEMA

            result["schema"] = LUNAR_ORBIT_SNAPSHOT_SCHEMA.json_schema()
        if self.kind == "opaque":
            result["opaquePayload"] = True
        return result


def scalar(kind="number", unit="1"):
    return ReportSchema(kind, unit=unit)


def array(item):
    return ReportSchema("array", item=item)


def identifiers(item):
    return ReportSchema("identifierMap", item=item)


def record(*groups, required=(), nullable=(), **children):
    fields = []
    for names, schema in (*groups, *((name, schema) for name, schema in children.items())):
        for name in names.split():
            if re.fullmatch(r"[a-z][A-Za-z0-9]*", name) is None:
                raise ValueError(f"Invalid public report field: {name!r}")
            fields.append(ReportField(name, schema, name in required, name in nullable))
    if len({f.name for f in fields}) != len(fields):
        raise ValueError("Duplicate report schema fields")
    if (set(required) | set(nullable)) - {f.name for f in fields}:
        raise ValueError("Undeclared report field constraint")
    return ReportSchema("object", tuple(fields))


NUMBER, INTEGER, BOOLEAN, STRING = (scalar(kind) for kind in ("number", "integer", "boolean", "string"))
METRES = scalar(unit="m")
SECONDS = scalar(unit="s")
OPAQUE = ReportSchema("opaque")
NUMBERS = identifiers(NUMBER)
STRINGS = array(STRING)
STATION_STATE = record(
    ("per", STRING),
    ("values", identifiers(METRES)),
    (
        "intervals",
        array(
            record(
                ("station start endExclusive name", STRING),
                required=("station", "start", "name"),
                nullable=("endExclusive",),
            )
        ),
    ),
    required=("per", "values"),
)
STATE = ReportSchema("parameterState")
DISTRIBUTION = record(
    ("count", INTEGER),
    (
        "rms median mad absoluteP50 absoluteP90 absoluteP95 absoluteP99 absoluteMaximum",
        NUMBER,
    ),
)
DISTRIBUTION_M = record(
    ("count", INTEGER),
    (
        "rms median mad absoluteP50 absoluteP90 absoluteP95 absoluteP99 absoluteMaximum",
        METRES,
    ),
)
WEIGHTS = record(
    (
        "observationCount fullWeightCount downweightedCount rejectedCount",
        INTEGER,
    ),
    ("minimum p05 median p95 maximum", NUMBER),
)
ACCURACY_GROUP = record(
    ("medianReportedSigmaM minimumValidSigmaM", METRES),
    ("observationCount rejectedCount retainedCount", INTEGER),
    nullable=("medianReportedSigmaM",),
)
ACCURACY_RECORD = record(
    ("componentId status reason", STRING),
    ("reportedSigmaM minimumValidSigmaM", METRES),
)
RESIDUAL_SCREEN = record(
    ("maximumAbsoluteM", METRES),
    ("maximumAbsoluteByStationM", identifiers(METRES)),
    nullable=("maximumAbsoluteM", "maximumAbsoluteByStationM"),
)
SIGMA_SCREEN = record(
    ("minimumOneWayM", METRES),
    ("minimumFractionOfGroupMedian", NUMBER),
)
SETTINGS = record(
    (
        "observationScreening",
        record(("residual", RESIDUAL_SCREEN), ("reportedSigma", SIGMA_SCREEN)),
    ),
    (
        "estimate",
        record(
            ("maxIterationCount", INTEGER),
            ("convergenceThresholdM", METRES),
            ("convergenceThresholdByParametrizationM", identifiers(METRES)),
            (
                "computeResiduals estimateVarianceFactors estimateRobustWeights",
                BOOLEAN,
            ),
            nullable=("convergenceThresholdByParametrizationM",),
        ),
    ),
    ("robustWeighting", record(("model", STRING), ("k0 k1", NUMBER), nullable=("k1",))),
    (
        "warmStartedSigmaFactorCount warmStartedWeightFactorCount",
        INTEGER,
    ),
)
SUMMARY = record(
    ("converged", BOOLEAN),
    ("terminationReason", STRING),
    (
        "sourceObservationCount initialLightTimeConvergedCount grossRejectedCount accuracyRejectedCount retainedObservationCount lastSolvedEquationCount equationEvaluationCount adjustmentIterationCount sigmaWeightIterationCount lastNormalMatrixRank finalEquationCount observationCount parameterCount rank",
        INTEGER,
    ),
    (
        "lastNormalMatrixCondition conditionNumber sigma0Post parameterUncertaintySigmaMultiplier",
        NUMBER,
    ),
    (
        "performanceSeconds",
        record(("cacheBuild normalSolve redundancy adjustSigma0", SECONDS)),
    ),
    nullable=("lastNormalMatrixCondition", "conditionNumber", "sigma0Post"),
)
EVALUATION = record(
    (
        "evaluation sourceObservationCount lightTimeConvergedCount lightTimeNonconvergedCount fixedDomainReturnedCount convergedButOutsideFixedDomainCount",
        INTEGER,
    ),
    ("purpose", STRING),
)
PARAMETER = record(
    ("name type maximumCorrelatedParameter unit", STRING),
    (
        "remainingLinearizedCorrectionM cofactorUncertaintyM formalUncertaintyM",
        METRES,
    ),
    ("maximumAbsoluteCorrelation", NUMBER),
    (
        "referenceValue appliedUpdate finalValue",
        scalar(unit="parameter.unit"),
    ),
    required=("name", "unit"),
    nullable=("maximumCorrelatedParameter", "maximumAbsoluteCorrelation", "formalUncertaintyM"),
)
VARIANCE = record(
    (
        "activeCount observationCount retainedObservationCount fullWeightCount downweightedCount rejectedCount",
        INTEGER,
    ),
    (
        "effectiveRedundancy weightedSquareSum sigmaFactorBefore sigmaFactorAfter varianceRatio initialSigmaFactor finalSigmaFactor varianceFactor standardizedRms medianStandardizedResidual madStandardizedResidual",
        NUMBER,
    ),
    (
        "updateStatus componentId configuredStart configuredEnd actualStartEpochUtc actualStartEpochLocal actualEndEpochUtc actualEndEpochLocal",
        STRING,
    ),
    ("proposedSigmaFactorApplied", BOOLEAN),
    ("residualRmsM residualWrmsM", METRES),
    ("reportedSigmaScreening", ACCURACY_GROUP),
    ("residualSummaryM", DISTRIBUTION_M),
    ("standardizedResidualSummary", DISTRIBUTION),
    (
        "weightFactorSummary finalStateProposedWeightFactorSummary",
        WEIGHTS,
    ),
    nullable=(
        "actualStartEpochUtc",
        "actualStartEpochLocal",
        "actualEndEpochUtc",
        "actualEndEpochLocal",
        "residualRmsM",
        "residualWrmsM",
        "standardizedRms",
        "medianStandardizedResidual",
        "madStandardizedResidual",
        "effectiveRedundancy",
        "weightedSquareSum",
        "varianceRatio",
        "sigmaFactorAfter",
    ),
)
SIGMA_ITERATION = record(
    (
        "iteration adjustmentIteration sigmaWeightIteration activeObservationCount rejectedObservationCount",
        INTEGER,
    ),
    ("elapsedSeconds", SECONDS),
    (
        "maximumSigmaFactorChange maximumWeightFactorChange totalFrozenRedundancy expectedTotalRedundancy normalMatrixCondition",
        NUMBER,
    ),
    ("candidateWrmsM maximumCandidateParameterUpdateM", METRES),
    ("candidateUpdateByBlockM", identifiers(METRES)),
    ("sigmaFactors", NUMBERS),
    ("weightFactorSummary", WEIGHTS),
    ("varianceComponents", identifiers(VARIANCE)),
    nullable=("candidateWrmsM", "normalMatrixCondition"),
)
ADJUSTMENT_ITERATION = record(
    (
        "iteration sigmaWeightIterations equationCount normalMatrixRank",
        INTEGER,
    ),
    ("maximumParameterUpdateM wrmsM", METRES),
    (
        "parameterUpdateWithinThreshold parameterConverged",
        BOOLEAN,
    ),
    (
        "convergenceThresholdByParametrizationM appliedUpdateByBlockM candidateUpdateByBlockM candidateParameterCorrectionsM",
        identifiers(METRES),
    ),
    (
        "normalizedParameterUpdateByBlock sigmaFactorsUsedInSolve sigmaFactorsForNextIteration",
        NUMBERS,
    ),
    ("normalMatrixCondition", NUMBER),
    (
        "weightFactorSummaryUsedInSolve weightFactorSummaryForNextIteration",
        WEIGHTS,
    ),
    ("stateAfterUpdate", STATE),
    nullable=("normalMatrixCondition",),
)
OBSERVATION = record(
    (
        "observationId epochUtc epochLocal stationId station varianceComponentId reportedSigmaScreeningStatus reportedSigmaScreeningReason appliedWeightStatus finalStateProposedWeightStatus",
        STRING,
    ),
    (
        "lightTimeConverged troposphereClamped proposedWeightFactorApplied",
        BOOLEAN,
    ),
    (
        "aprioriSigmaM minimumValidSigmaM sigma0M currentStateResidualM linearizedPostfitResidualM residualSigmaM",
        METRES,
    ),
    ("baseVarianceM2", scalar(unit="m^2")),
    ("baseWeightPerM2 equivalentWeightPerM2", scalar(unit="m^-2")),
    (
        "sigmaFactor leverage standardizedResidual appliedWeightFactor finalStateProposedWeightFactor",
        NUMBER,
    ),
    ("matchedParameterNames", STRINGS),
    required=("observationId", "stationId", "epochUtc", "lightTimeConverged", "aprioriSigmaM"),
    nullable=(
        "currentStateResidualM",
        "linearizedPostfitResidualM",
        "residualSigmaM",
        "standardizedResidual",
        "leverage",
        "reportedSigmaScreeningReason",
    ),
)
PROCESSING_STEP = record(
    (
        "type name outputFile outputFileReport outputFileState outputFileSolution outputFileCovariance outputFileReflectorCatalog outputLevel",
        STRING,
    ),
    ("parametrizations", STRINGS),
    ("retainedCount", INTEGER),
    ("settings", SETTINGS),
    ("summary", SUMMARY),
    ("state", STATE),
    (
        "residual",
        record(
            ("maximumAbsoluteM", METRES),
            ("maximumAbsoluteByStationM", identifiers(METRES)),
            ("rejectedCount", INTEGER),
            nullable=("maximumAbsoluteM", "maximumAbsoluteByStationM"),
        ),
    ),
    (
        "reportedSigma",
        record(("minimumOneWayM", METRES), ("minimumFractionOfGroupMedian", NUMBER), ("rejectedCount", INTEGER)),
    ),
    nullable=(
        "outputFileReport",
        "outputFileState",
        "outputFileSolution",
        "outputFileCovariance",
        "outputFileReflectorCatalog",
    ),
)
KERNEL_ENTRY = record(
    ("relativePath sha256", STRING), ("loadOrder", INTEGER), required=("relativePath", "sha256", "loadOrder")
)
SCIENTIFIC_FILE = record(("path sha256", STRING), required=("path", "sha256"))
KERNEL_DIRECTORY = record(("path", STRING), ("kernels", array(KERNEL_ENTRY)), required=("path", "kernels"))
GRAVITY_SOURCE = record(
    ("path norm tideSystem gmSource radiusSource", STRING),
    ("declaredDegree loadedDegree", INTEGER),
    ("allowMissingCoefficients", BOOLEAN),
    ("missingCoefficients", array(array(INTEGER))),
    ("fileGmM3S2", scalar(unit="m^3/s^2")),
    ("fileRadiusM", METRES),
    required=(
        "path",
        "norm",
        "tideSystem",
        "declaredDegree",
        "loadedDegree",
        "allowMissingCoefficients",
        "missingCoefficients",
        "gmSource",
        "radiusSource",
        "fileGmM3S2",
        "fileRadiusM",
    ),
)
PROCESSING_REPORT_SCHEMA = record(
    ("converged", BOOLEAN),
    (
        "publicationQuality",
        record(
            ("formalProductEligible", BOOLEAN),
            ("status", STRING),
            ("reasons", STRINGS),
            required=("formalProductEligible", "status", "reasons"),
        ),
    ),
    ("terminationReason fingerprint", STRING),
    ("settings", SETTINGS),
    ("equationEvaluations", array(EVALUATION)),
    ("parameterNames", STRINGS),
    ("state", STATE),
    (
        "observationScreening",
        record(
            (
                "residual",
                record(
                    ("rejectedCount", INTEGER),
                    ("rejectedObservations", identifiers(METRES)),
                ),
            ),
            (
                "reportedSigma",
                record(
                    ("action", STRING),
                    ("minimumOneWayM", METRES),
                    ("minimumFractionOfGroupMedian", NUMBER),
                    ("rejectedCount", INTEGER),
                    ("rejectedObservations", identifiers(ACCURACY_RECORD)),
                    ("groups", identifiers(ACCURACY_GROUP)),
                ),
            ),
        ),
    ),
    ("sigmaFactors weightFactors", NUMBERS),
    ("sigmaWeightIterations", array(SIGMA_ITERATION)),
    ("adjustmentIterations", array(ADJUSTMENT_ITERATION)),
    ("summary", SUMMARY),
    ("parameters", array(PARAMETER)),
    (
        "globalResiduals",
        record(
            ("residualM", DISTRIBUTION_M),
            ("standardizedResidual", DISTRIBUTION),
            ("equivalentWeightedRmsM", METRES),
            ("weightFactors", WEIGHTS),
            nullable=("equivalentWeightedRmsM",),
        ),
    ),
    ("varianceComponents", array(VARIANCE)),
    ("observations", array(OBSERVATION)),
    (
        "scientificInputs",
        record(
            ("resources", identifiers(array(ReportSchema("scientificResource")))),
            ("codeSha256", STRING),
            ("versions", identifiers(STRING)),
            required=("resources", "codeSha256", "versions"),
        ),
    ),
    ("execution", record(("mode", STRING), ("workerCount", INTEGER))),
    ("processingSteps", array(PROCESSING_STEP)),
    ("finalRemainingCorrection", identifiers(METRES)),
    required=(
        "converged",
        "publicationQuality",
        "terminationReason",
        "settings",
        "summary",
        "parameters",
        "observations",
    ),
)
LUNAR_ORBIT_METADATA_SCHEMA = record(
    (
        "model timeScale earthPoleModel frame units solarGmModel kuiperBeltRingModel scipyVersion integrator startupIntegrator lunarAttitudeMode tideHistory orbitFile accelerationDiagnosticsFile rotationFile",
        STRING,
    ),
    ("config", ReportSchema("configuration")),
    ("resolvedForceInventory", record(("active", STRINGS), ("source", STRING), required=("active", "source"))),
    ("resolvedGravityFields", identifiers(GRAVITY_SOURCE)),
    (
        "initialStateSi",
        array(scalar(unit="mixed: relative position m, velocity m/s, quaternion 1, angular velocity rad/s")),
    ),
    ("dynamicsBodyNames solarForceTerms", STRINGS),
    ("kernels", array(KERNEL_ENTRY)),
    (
        "stepSeconds startupStepSeconds rotationSeedDurationSeconds earthTideMaxTranslationExtrapolationSeconds",
        SECONDS,
    ),
    (
        "earthTideAdvancedTranslationQueries orbitSamples accelerationDiagnosticSamples",
        INTEGER,
    ),
    (
        "referenceDifferenceRtnRmsM referenceDifferenceRtnMaxAbsM",
        array(METRES),
    ),
    required=("model", "timeScale", "config"),
    nullable=(
        "kuiperBeltRingModel",
        "rotationSeedDurationSeconds",
        "earthTideAdvancedTranslationQueries",
        "earthTideMaxTranslationExtrapolationSeconds",
        "rotationFile",
    ),
)
