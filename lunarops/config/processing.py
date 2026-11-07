"""Config schemas of the LLR processing plan.

The plan of the ``LlrProcessing`` program consists of ``varianceComponents`` (observation groups with
their own sigma factor, selected by station, UTC date interval and laser wavelength) and an ordered list of
``processingSteps``: one ``screenObservations`` step, ``selectParametrizations`` and ``estimate`` steps, and
the output steps ``writeResiduals``, ``writeNormalEquations`` and ``writeResults``.
"""

from datetime import date

from lunarops.estimation.robust_weights import ROBUST_WEIGHT_MODELS

from .schema import ConfigSchema, boolean, integer, mapping, number, path, sequence, string, time


def _numbers(config, location, *, nullable=False):
    spec = number("value", ge=0, allow_none=nullable)
    for key, value in config.items():
        string("key").validate(key, location)
        config[key] = spec.validate(value, f"{location}.{key}")
    return config


def _robust(config, location):
    model = config["model"]
    if model == "igg3":
        config.setdefault("k1", 6.0)
        if not 0 < config["k0"] < config["k1"]:
            raise ValueError(f"{location}: IGGIII thresholds must satisfy 0 < k0 < k1.")
    elif "k1" in config:
        raise ValueError(f"{location}: directRejection uses k0 only; omit k1.")
    return config


ROBUST = ConfigSchema(
    (
        string(
            "model",
            default="igg3",
            choices=tuple(ROBUST_WEIGHT_MODELS),
            description=(
                "Robust weighting function: `igg3` (full weight below `k0`, down-weighted between `k0` and `k1`, zero"
                " above `k1`) or `directRejection` (zero weight above `k0`)."
            ),
        ),
        number(
            "k0",
            default=1.5,
            gt=0,
            description="Lower threshold of the robust weighting function, in units of the standardized residual.",
        ),
        number(
            "k1",
            gt=0,
            description=(
                "Upper threshold of `igg3`, in units of the standardized residual; must exceed `k0` (6 if omitted). "
                "Not used by `directRejection`."
            ),
        ),
    ),
    validator=_robust,
)


def _residual(config, location):
    config["maximumAbsoluteByStationM"] = _numbers(config["maximumAbsoluteByStationM"], location, nullable=True)
    return config


RESIDUAL = ConfigSchema(
    (
        number(
            "maximumAbsoluteM",
            default=20.0,
            ge=0,
            allow_none=True,
            description="Rejection limit for the absolute prefit one-way residual, in metres. Null disables the test.",
        ),
        mapping(
            "maximumAbsoluteByStationM",
            default={},
            description=(
                "Per-station replacement of `maximumAbsoluteM`, keyed by station catalog key, in metres. A null value"
                " disables the test for that station."
            ),
        ),
    ),
    validator=_residual,
)
ACCURACY = ConfigSchema(
    (
        number(
            "minimumOneWayM",
            default=1e-3,
            ge=0,
            description=(
                "Smallest plausible reported one-way sigma, in metres. Normal points reporting a smaller sigma are "
                "rejected; retained sigmas are not changed."
            ),
        ),
        number(
            "minimumFractionOfGroupMedian",
            default=0.1,
            ge=0,
            le=1,
            description=(
                "Reported sigmas below this fraction of the median sigma of their variance component are rejected. "
                "The effective limit is the larger of this and `minimumOneWayM`."
            ),
        ),
    )
)


def _estimate(config, location):
    if not config["computeResiduals"] and (config["estimateVarianceFactors"] or config["estimateRobustWeights"]):
        raise ValueError(
            f"{location}: estimateVarianceFactors and estimateRobustWeights require computeResiduals=true."
        )
    if config["sessionCorrelation"] and config["varianceEstimation"] != "consistent":
        raise ValueError(f"{location}: sessionCorrelation requires varianceEstimation=consistent.")
    if config["varianceModel"] != "scaled" and config["varianceEstimation"] != "consistent":
        raise ValueError(f"{location}: varianceModel={config['varianceModel']} requires varianceEstimation=consistent.")
    if config["earthOrientationCorrection"] and config["varianceEstimation"] != "consistent":
        raise ValueError(f"{location}: earthOrientationCorrection requires varianceEstimation=consistent.")
    config["convergenceThresholdByParametrizations"] = _numbers(
        config["convergenceThresholdByParametrizations"], location
    )
    return config


def _selectors(config, location):
    values = config["parametrizations"]
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
            (
                mapping(
                    "residual",
                    default={},
                    nested=RESIDUAL,
                    description="Prefit residual screening limits.",
                ),
                mapping(
                    "reportedSigma",
                    default={},
                    nested=ACCURACY,
                    description="Screening limits for the reported normal-point sigma.",
                ),
            ),
            type_name="screenObservations",
        ),
        ConfigSchema(
            (
                sequence(
                    "parametrizations",
                    required=True,
                    item_kind="string",
                    non_empty=True,
                    min_items=1,
                    description=(
                        "Parametrization types estimated by the following `estimate` steps, replacing the previous "
                        "selection."
                    ),
                ),
            ),
            type_name="selectParametrizations",
            validator=_selectors,
        ),
        ConfigSchema(
            (
                string(
                    "name",
                    required=True,
                    description="Unique name of the estimate step; it labels the step in reports.",
                ),
                integer(
                    "maxIterationCount",
                    default=3,
                    ge=1,
                    description="Maximum number of linearization iterations of the nonlinear adjustment.",
                ),
                number(
                    "convergenceThreshold",
                    default=1e-2,
                    ge=0,
                    description=(
                        "The step converges when the largest update of every parameter block is at most this value, "
                        "in the block's unit (metres for positions and biases, using the 3-D norm per reflector or "
                        "station)."
                    ),
                ),
                mapping(
                    "convergenceThresholdByParametrizations",
                    default={},
                    description=(
                        "Per-block replacement of `convergenceThreshold`, keyed by parametrization type, in the "
                        "block's unit."
                    ),
                ),
                boolean(
                    "computeResiduals",
                    default=True,
                    description=(
                        "Evaluate postfit residuals after the step. Required by variance-factor and robust-weight "
                        "estimation."
                    ),
                ),
                boolean(
                    "estimateVarianceFactors",
                    default=True,
                    description=(
                        "Estimate one sigma factor per variance component from the postfit residuals and redundancies."
                    ),
                ),
                boolean(
                    "estimateRobustWeights",
                    default=True,
                    description="Estimate per-observation weight factors with the `robustWeighting` function.",
                ),
                mapping(
                    "robustWeighting",
                    default={},
                    nested=ROBUST,
                    description="Robust weighting function and thresholds.",
                ),
                string(
                    "varianceEstimation",
                    default="consistent",
                    choices=("consistent", "legacy"),
                    description=(
                        "`consistent`: at every linearization, iterate the solution, the variance factors (divided by "
                        "the Gaussian consistency factor of the robust weight function) and the robust weights "
                        "(residuals standardized by their variance under the current weights) to convergence. `legacy`: one solution per "
                        "linearization followed by ten alternating variance/weight updates of its residuals."
                    ),
                ),
                boolean(
                    "sessionCorrelation",
                    default=False,
                    description=(
                        "Model one random offset per night session of each variance component (correlated normal "
                        "points of one night) and estimate its sigma per component. Requires `consistent`."
                    ),
                ),
                number(
                    "sessionMaximumGapHours",
                    default=4.0,
                    gt=0,
                    description="Largest gap between consecutive normal points of one session, in hours.",
                ),
                number(
                    "sessionInitialSigmaM",
                    default=0.005,
                    gt=0,
                    description="Starting sigma of the session offsets, in metres.",
                ),
                string(
                    "varianceModel",
                    default="scaled",
                    choices=("scaled", "floorAndScaled"),
                    description=(
                        "Variance function of each variance component. `scaled`: sigma = s * reported sigma. "
                        "`floorAndScaled`: sigma^2 = a^2 + (s * reported sigma)^2, an error floor `a` added in "
                        "quadrature, with `a` and `s` estimated by least-squares variance component estimation. "
                        "Requires `consistent`; a `scaled` estimate drops floors of earlier estimates."
                    ),
                ),
                boolean(
                    "earthOrientationCorrection",
                    default=False,
                    description=(
                        "Estimate corrections of UT1 and the polar motion per station night (normal points of one "
                        "station without a gap longer than `sessionMaximumGapHours`) as random effects, with the "
                        "uncertainties of `inputFileEarthOrientationUncertainty` as prior sigmas. Requires "
                        "`consistent` and the station position partials (a `stationPosition` parametrization)."
                    ),
                ),
                number(
                    "earthOrientationSigmaScale",
                    default=1.0,
                    gt=0,
                    description="Factor applied to the tabulated Earth-orientation uncertainties.",
                ),
            ),
            type_name="estimate",
            validator=_estimate,
        ),
        ConfigSchema(
            (
                path(
                    "outputFile",
                    required=True,
                    non_empty=True,
                    description="Output observation-result file with the final postfit residuals.",
                ),
                string(
                    "outputLevel",
                    default="standard",
                    choices=("standard", "full"),
                    description="Column set of the residual file: `standard` or `full`.",
                ),
            ),
            type_name="writeResiduals",
        ),
        ConfigSchema(
            (
                path(
                    "outputFile",
                    required=True,
                    non_empty=True,
                    description="Output directory for the final normal equations.",
                ),
            ),
            type_name="writeNormalEquations",
        ),
        ConfigSchema(
            tuple(
                path(key, non_empty=True, allow_none=True, description=description)
                for key, description in (
                    ("outputFileReport", "Output processing report: convergence, statistics and publication quality."),
                    (
                        "outputFileState",
                        "Output processing state for continuing the run with inputFileProcessingState.",
                    ),
                    ("outputFileSolution", "Output parameter vector with final values and formal uncertainties."),
                    ("outputFileCovariance", "Output directory with the parameter covariance matrix."),
                    ("outputFileReflectorCatalog", "Output reflector catalog updated with the estimated positions."),
                    ("outputFileStationCatalog", "Output station catalog updated with the estimated positions."),
                )
            ),
            type_name="writeResults",
            validator=_outputs,
        ),
    )
)


def _component(config, location):
    for key in ("start", "endExclusive"):
        if config.get(key) is not None:
            config[key] = date.fromisoformat(str(config[key])).isoformat()
    if config.get("endExclusive") is not None and config["endExclusive"] <= config["start"]:
        raise ValueError(f"{location}: endExclusive must be after start.")
    low, high = config.get("wavelengthMinNm"), config.get("wavelengthMaxExclusiveNm")
    if low is not None and high is not None and low >= high:
        raise ValueError(f"{location}: wavelength range is empty.")
    return config


COMPONENT = ConfigSchema(
    (
        string("id", required=True, description="Unique identifier of the variance component."),
        string(
            "station",
            required=True,
            description="Station catalog key whose normal points belong to this component.",
        ),
        time("start", required=True, description="First UTC date of the component (inclusive)."),
        time(
            "endExclusive",
            allow_none=True,
            description="UTC date at which the component ends (exclusive). Null means open-ended.",
        ),
        number(
            "wavelengthMinNm",
            gt=0,
            allow_none=True,
            description=(
                "Lower laser-wavelength bound of the component (inclusive), in nanometres. Null means no bound."
            ),
        ),
        number(
            "wavelengthMaxExclusiveNm",
            gt=0,
            allow_none=True,
            description=(
                "Upper laser-wavelength bound of the component (exclusive), in nanometres. Null means no bound."
            ),
        ),
    ),
    validator=_component,
)
def _minimal_constraints(config, location):
    conditions = config["conditions"]
    allowed = ("translation", "rotation", "scale")
    if not conditions or len(set(conditions)) != len(conditions) or not set(conditions) <= set(allowed):
        raise ValueError(f"{location}.conditions must be distinct items of {list(allowed)}.")
    return config


CONSTRAINT_SCHEMA = ConfigSchema(
    variants=(
        ConfigSchema(
            (
                sequence(
                    "parameters",
                    required=True,
                    item_kind="string",
                    min_items=1,
                    non_empty=True,
                    description=(
                        "Patterns (`fnmatch`) of the constrained parameter names `object:type:temporal:interval`, "
                        "e.g. `*:rangeBias*` or `APOLLO:position.*`."
                    ),
                ),
                number("sigma", required=True, gt=0, description="A-priori sigma, in the unit of the parameters."),
            ),
            type_name="aprioriSigma",
        ),
        ConfigSchema(
            (
                sequence(
                    "stations",
                    item_kind="string",
                    unique=True,
                    min_items=3,
                    non_empty=True,
                    description=(
                        "Datum stations (at least three). Defaults to every station whose position is estimated."
                    ),
                ),
                sequence(
                    "conditions",
                    item_kind="string",
                    default=["translation", "rotation"],
                    description=(
                        "Helmert parameters of the datum stations constrained to zero: `translation` "
                        "(no-net-translation), `rotation` (no-net-rotation) and `scale`."
                    ),
                ),
                number(
                    "sigmaM",
                    default=1.0e-4,
                    gt=0,
                    description=(
                        "Sigma of each constrained Helmert parameter, in metres; rotations and scale are expressed "
                        "as displacements at the Earth's equatorial radius."
                    ),
                ),
            ),
            type_name="minimalConstraints",
            validator=_minimal_constraints,
        ),
    )
)


PROCESSING_FIELDS = (
    sequence(
        "varianceComponents",
        required=True,
        item_kind="mapping",
        item_nested=COMPONENT,
        min_items=1,
        description=(
            "Observation groups that each receive their own sigma factor. Every retained normal point must match "
            "exactly one component."
        ),
    ),
    sequence(
        "processingSteps",
        required=True,
        item_kind="mapping",
        item_nested=STEP_SCHEMA,
        min_items=1,
        description=(
            "Ordered processing plan: one leading `screenObservations`, any `selectParametrizations` and `estimate` "
            "steps, then one or more output steps."
        ),
    ),
    sequence(
        "constraints",
        default=[],
        item_kind="mapping",
        item_nested=CONSTRAINT_SCHEMA,
        description=(
            "Parameter constraints, added as weighted pseudo-observations to every estimate that has their "
            "parameters: `aprioriSigma` ties parameters to their a-priori values, `minimalConstraints` defines the "
            "station datum by minimum constraints to the a-priori coordinates (instead of fixed stations)."
        ),
    ),
)


def validate_processing_plan(config, location):
    """Check the order of the processing steps and the uniqueness of names and IDs.

    The plan must start with exactly one ``screenObservations`` step, contain at least one ``estimate``
    step, have distinct ``estimate`` names and variance-component IDs, and end with one or more output
    steps that follow every selection and estimate step.

    Parameters
    ----------
    config : dict
        Program config with resolved ``processingSteps`` and ``varianceComponents``.
    location : str
        Config path used in error messages.

    Returns
    -------
    dict
        The config, unchanged.

    Raises
    ------
    ValueError
        If the plan violates one of the rules.
    """
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
