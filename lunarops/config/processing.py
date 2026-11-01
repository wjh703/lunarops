"""Native configuration schemas for LLR processing stages."""

from datetime import date

from lunarops.base.station_identity import validate_station_id
from lunarops.estimation.robust_weights import ROBUST_WEIGHT_MODELS

from .schema import ConfigSchema, boolean, integer, mapping, number, path, sequence, string, time


def _numbers(config, location, *, nullable=False):
    spec = number("value", ge=0, allow_none=nullable)
    for key, value in config.items():
        string("key").validate(key, location)
        config[key] = None if value is None else float(spec.validate(value, f"{location}.{key}"))
        if value is None and not nullable:
            spec.validate(value, f"{location}.{key}")
    return config


def _robust(config, location):
    model = config["model"].strip()
    config["model"] = model
    if model == "igg3":
        config.setdefault("k1", 6.0)
        if not 0 < config["k0"] < config["k1"]:
            raise ValueError(f"{location}: IGGIII thresholds must satisfy 0 < k0 < k1.")
    elif "k1" in config:
        raise ValueError(f"{location}: directRejection uses k0 only; omit k1.")
    config["k0"] = float(config["k0"])
    if "k1" in config:
        config["k1"] = float(config["k1"])
    return config


ROBUST = ConfigSchema(
    (
        string("model", default="igg3", choices=tuple(ROBUST_WEIGHT_MODELS)),
        number("k0", default=1.5, gt=0),
        number("k1", gt=0),
    ),
    validator=_robust,
)


def _residual(config, location):
    config["maximumAbsoluteByStationM"] = _numbers(config["maximumAbsoluteByStationM"], location, nullable=True)
    return config


RESIDUAL = ConfigSchema(
    (number("maximumAbsoluteM", default=20.0, ge=0, allow_none=True), mapping("maximumAbsoluteByStationM", default={})),
    validator=_residual,
)
ACCURACY = ConfigSchema(
    (number("minimumOneWayM", default=1e-3, ge=0), number("minimumFractionOfGroupMedian", default=0.1, ge=0, le=1))
)


def _estimate(config, location):
    if not config["computeResiduals"] and (config["estimateVarianceFactors"] or config["estimateRobustWeights"]):
        raise ValueError(
            f"{location}: estimateVarianceFactors and estimateRobustWeights require computeResiduals=true."
        )
    config["convergenceThresholdByParametrizations"] = _numbers(
        config["convergenceThresholdByParametrizations"], location
    )
    config["name"] = config["name"].strip()
    return config


def _selectors(config, location):
    values = [value.strip() for value in config["parametrizations"]]
    if len(set(values)) != len(values):
        raise ValueError(f"{location}: Selected parametrizations must be unique.")
    config["parametrizations"] = values
    return config


def _outputs(config, location):
    if not any(value is not None for key, value in config.items() if key != "type"):
        raise ValueError(f"{location}: writeResults requires at least one output file.")
    return config


STEP_SCHEMA = ConfigSchema(
    variants=(
        ConfigSchema(
            (mapping("residual", default={}, nested=RESIDUAL), mapping("reportedSigma", default={}, nested=ACCURACY)),
            type_name="screenObservations",
        ),
        ConfigSchema(
            (sequence("parametrizations", required=True, item_kind="string", non_empty=True, min_items=1),),
            type_name="selectParametrizations",
            validator=_selectors,
        ),
        ConfigSchema(
            (
                string("name", required=True),
                integer("maxIterationCount", default=3, ge=1),
                number("convergenceThreshold", default=1e-2, ge=0),
                mapping(
                    "convergenceThresholdByParametrizations",
                    default={},
                ),
                boolean("computeResiduals", default=True),
                boolean("estimateVarianceFactors", default=True),
                boolean("estimateRobustWeights", default=True),
                mapping("robustWeighting", default={}, nested=ROBUST),
            ),
            type_name="estimate",
            validator=_estimate,
        ),
        ConfigSchema(
            (
                path("outputFile", required=True, non_empty=True),
                string("outputLevel", default="standard", choices=("standard", "full")),
            ),
            type_name="writeResiduals",
        ),
        ConfigSchema((path("outputFile", required=True, non_empty=True),), type_name="writeNormalEquations"),
        ConfigSchema(
            tuple(
                path(key, non_empty=True, allow_none=True)
                for key in (
                    "outputFileReport",
                    "outputFileState",
                    "outputFileSolution",
                    "outputFileCovariance",
                    "outputFileReflectorCatalog",
                    "outputFileStationCatalog",
                )
            ),
            type_name="writeResults",
            validator=_outputs,
        ),
    )
)


def _component(config, location):
    config["id"] = config["id"].strip()
    config["station"] = validate_station_id(config["station"])
    for key in ("start", "endExclusive"):
        if config.get(key) is not None:
            config[key] = date.fromisoformat(str(config[key])).isoformat()
    if config.get("endExclusive") is not None and config["endExclusive"] <= config["start"]:
        raise ValueError(f"{location}: endExclusive must be after start.")
    low, high = config.get("wavelengthMinNm"), config.get("wavelengthMaxExclusiveNm")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"{location}: wavelength range is empty.")
    for key in ("wavelengthMinNm", "wavelengthMaxExclusiveNm"):
        if config.get(key) is not None:
            config[key] = float(config[key])
    return config


COMPONENT = ConfigSchema(
    (
        string("id", required=True),
        string("station", required=True),
        time("start", required=True),
        time("endExclusive", allow_none=True),
        number("wavelengthMinNm", gt=0, allow_none=True),
        number("wavelengthMaxExclusiveNm", gt=0, allow_none=True),
    ),
    validator=_component,
)
PROCESSING_FIELDS = (
    sequence("varianceComponents", required=True, item_kind="mapping", item_nested=COMPONENT, min_items=1),
    sequence("processingSteps", required=True, item_kind="mapping", item_nested=STEP_SCHEMA, min_items=1),
)


def validate_processing_plan(config, location):
    steps = config["processingSteps"]
    types = [step["type"] for step in steps]
    if types.count("screenObservations") != 1 or types[0] != "screenObservations":
        raise ValueError(f"{location}: Processing plan must start with exactly one screenObservations step.")
    estimates = [step["name"] for step in steps if step["type"] == "estimate"]
    if not estimates:
        raise ValueError(f"{location}: Adjustment plan must contain at least one estimate step.")
    if len(set(estimates)) != len(estimates):
        raise ValueError(f"{location}: Estimate step names must be unique.")
    writers = {"writeResiduals", "writeNormalEquations", "writeResults"}
    if not writers.intersection(types):
        raise ValueError(f"{location}: Processing plan must contain at least one output step.")
    first = next(index for index, typ in enumerate(types) if typ in writers)
    if any(typ not in writers for typ in types[first:]):
        raise ValueError(f"{location}: Selection and estimate steps must precede all output steps.")
    ids = [component["id"] for component in config["varianceComponents"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{location}: Variance-component IDs must be unique.")
    return config


PROCESSING_SCHEMA = ConfigSchema(PROCESSING_FIELDS, validator=validate_processing_plan)
