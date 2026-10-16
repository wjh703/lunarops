"""One-way serialization of current adjustment implementation objects.

These private record builders describe current solver data, not input formats.
Readers accept only the independently declared canonical report schema.
"""

from dataclasses import asdict

from lunarops.base.parameter_name import names_to_strings
from lunarops.base.serialization import plain_data


def _interval(value):
    result = {}
    if "station" in value:
        result["station"] = None if value["station"] is None else plain_data(value["station"])
    if "start" in value:
        result["start"] = None if value["start"] is None else plain_data(value["start"])
    if "end_exclusive" in value:
        result["endExclusive"] = None if value["end_exclusive"] is None else plain_data(value["end_exclusive"])
    if "name" in value:
        result["name"] = None if value["name"] is None else plain_data(value["name"])
    return result


def _station_state(value):
    result = {}
    if "per" in value:
        result["per"] = None if value["per"] is None else plain_data(value["per"])
    if "values" in value:
        result["values"] = (
            None if value["values"] is None else {str(key): plain_data(item) for key, item in value["values"].items()}
        )
    if "intervals" in value:
        result["intervals"] = None if value["intervals"] is None else [_interval(item) for item in value["intervals"]]
    return result


def _residual_screen(value):
    result = {}
    if "maximum_absolute_m" in value:
        result["maximumAbsoluteM"] = (
            None if value["maximum_absolute_m"] is None else plain_data(value["maximum_absolute_m"])
        )
    if "maximum_absolute_by_station_m" in value:
        result["maximumAbsoluteByStationM"] = (
            None
            if value["maximum_absolute_by_station_m"] is None
            else {str(key): plain_data(item) for key, item in value["maximum_absolute_by_station_m"].items()}
        )
    return result


def _sigma_screen(value):
    result = {}
    if "minimum_one_way_m" in value:
        result["minimumOneWayM"] = (
            None if value["minimum_one_way_m"] is None else plain_data(value["minimum_one_way_m"])
        )
    if "minimum_fraction_of_group_median" in value:
        result["minimumFractionOfGroupMedian"] = (
            None
            if value["minimum_fraction_of_group_median"] is None
            else plain_data(value["minimum_fraction_of_group_median"])
        )
    return result


def _screening_settings(value):
    result = {}
    if "residual" in value:
        result["residual"] = None if value["residual"] is None else _residual_screen(value["residual"])
    if "reported_sigma" in value:
        result["reportedSigma"] = None if value["reported_sigma"] is None else _sigma_screen(value["reported_sigma"])
    return result


def _estimate_settings(value):
    result = {}
    if "max_iteration_count" in value:
        result["maxIterationCount"] = (
            None if value["max_iteration_count"] is None else plain_data(value["max_iteration_count"])
        )
    if "convergence_threshold_m" in value:
        result["convergenceThresholdM"] = (
            None if value["convergence_threshold_m"] is None else plain_data(value["convergence_threshold_m"])
        )
    if "convergence_threshold_by_parametrization_m" in value:
        result["convergenceThresholdByParametrizationM"] = (
            None
            if value["convergence_threshold_by_parametrization_m"] is None
            else {
                str(key): plain_data(item) for key, item in value["convergence_threshold_by_parametrization_m"].items()
            }
        )
    if "compute_residuals" in value:
        result["computeResiduals"] = (
            None if value["compute_residuals"] is None else plain_data(value["compute_residuals"])
        )
    if "estimate_variance_factors" in value:
        result["estimateVarianceFactors"] = (
            None if value["estimate_variance_factors"] is None else plain_data(value["estimate_variance_factors"])
        )
    if "estimate_robust_weights" in value:
        result["estimateRobustWeights"] = (
            None if value["estimate_robust_weights"] is None else plain_data(value["estimate_robust_weights"])
        )
    return result


def _robust_settings(value):
    result = {}
    if "model" in value:
        result["model"] = None if value["model"] is None else plain_data(value["model"])
    if "k0" in value:
        result["k0"] = None if value["k0"] is None else plain_data(value["k0"])
    if "k1" in value:
        result["k1"] = None if value["k1"] is None else plain_data(value["k1"])
    return result


def serialize_settings(value):
    result = {}
    if "observation_screening" in value:
        result["observationScreening"] = (
            None if value["observation_screening"] is None else _screening_settings(value["observation_screening"])
        )
    if "estimate" in value:
        result["estimate"] = None if value["estimate"] is None else _estimate_settings(value["estimate"])
    if "robust_weighting" in value:
        result["robustWeighting"] = (
            None if value["robust_weighting"] is None else _robust_settings(value["robust_weighting"])
        )
    if "warm_started_sigma_factor_count" in value:
        result["warmStartedSigmaFactorCount"] = (
            None
            if value["warm_started_sigma_factor_count"] is None
            else plain_data(value["warm_started_sigma_factor_count"])
        )
    if "warm_started_weight_factor_count" in value:
        result["warmStartedWeightFactorCount"] = (
            None
            if value["warm_started_weight_factor_count"] is None
            else plain_data(value["warm_started_weight_factor_count"])
        )
    return result


def _evaluation(value):
    result = {}
    if "evaluation" in value:
        result["evaluation"] = None if value["evaluation"] is None else plain_data(value["evaluation"])
    if "source_observation_count" in value:
        result["sourceObservationCount"] = (
            None if value["source_observation_count"] is None else plain_data(value["source_observation_count"])
        )
    if "light_time_converged_count" in value:
        result["lightTimeConvergedCount"] = (
            None if value["light_time_converged_count"] is None else plain_data(value["light_time_converged_count"])
        )
    if "light_time_nonconverged_count" in value:
        result["lightTimeNonconvergedCount"] = (
            None
            if value["light_time_nonconverged_count"] is None
            else plain_data(value["light_time_nonconverged_count"])
        )
    if "fixed_domain_returned_count" in value:
        result["fixedDomainReturnedCount"] = (
            None if value["fixed_domain_returned_count"] is None else plain_data(value["fixed_domain_returned_count"])
        )
    if "converged_but_outside_fixed_domain_count" in value:
        result["convergedButOutsideFixedDomainCount"] = (
            None
            if value["converged_but_outside_fixed_domain_count"] is None
            else plain_data(value["converged_but_outside_fixed_domain_count"])
        )
    if "purpose" in value:
        result["purpose"] = None if value["purpose"] is None else plain_data(value["purpose"])
    return result


def _rejected(value):
    result = {}
    if "rejected_count" in value:
        result["rejectedCount"] = None if value["rejected_count"] is None else plain_data(value["rejected_count"])
    if "rejected_observations" in value:
        result["rejectedObservations"] = (
            None
            if value["rejected_observations"] is None
            else {str(key): plain_data(item) for key, item in value["rejected_observations"].items()}
        )
    return result


def _accuracy_record(value):
    result = {}
    if "component_id" in value:
        result["componentId"] = None if value["component_id"] is None else plain_data(value["component_id"])
    if "status" in value:
        result["status"] = None if value["status"] is None else plain_data(value["status"])
    if "reason" in value:
        result["reason"] = None if value["reason"] is None else plain_data(value["reason"])
    if "reported_sigma_m" in value:
        result["reportedSigmaM"] = None if value["reported_sigma_m"] is None else plain_data(value["reported_sigma_m"])
    if "minimum_valid_sigma_m" in value:
        result["minimumValidSigmaM"] = (
            None if value["minimum_valid_sigma_m"] is None else plain_data(value["minimum_valid_sigma_m"])
        )
    return result


def _accuracy_group(value):
    result = {}
    if "median_reported_sigma_m" in value:
        result["medianReportedSigmaM"] = (
            None if value["median_reported_sigma_m"] is None else plain_data(value["median_reported_sigma_m"])
        )
    if "minimum_valid_sigma_m" in value:
        result["minimumValidSigmaM"] = (
            None if value["minimum_valid_sigma_m"] is None else plain_data(value["minimum_valid_sigma_m"])
        )
    if "observation_count" in value:
        result["observationCount"] = (
            None if value["observation_count"] is None else plain_data(value["observation_count"])
        )
    if "rejected_count" in value:
        result["rejectedCount"] = None if value["rejected_count"] is None else plain_data(value["rejected_count"])
    if "retained_count" in value:
        result["retainedCount"] = None if value["retained_count"] is None else plain_data(value["retained_count"])
    return result


def _accuracy(value):
    result = {}
    if "action" in value:
        result["action"] = None if value["action"] is None else plain_data(value["action"])
    if "minimum_one_way_m" in value:
        result["minimumOneWayM"] = (
            None if value["minimum_one_way_m"] is None else plain_data(value["minimum_one_way_m"])
        )
    if "minimum_fraction_of_group_median" in value:
        result["minimumFractionOfGroupMedian"] = (
            None
            if value["minimum_fraction_of_group_median"] is None
            else plain_data(value["minimum_fraction_of_group_median"])
        )
    if "rejected_count" in value:
        result["rejectedCount"] = None if value["rejected_count"] is None else plain_data(value["rejected_count"])
    if "rejected_observations" in value:
        result["rejectedObservations"] = (
            None
            if value["rejected_observations"] is None
            else {str(key): _accuracy_record(item) for key, item in value["rejected_observations"].items()}
        )
    if "groups" in value:
        result["groups"] = (
            None
            if value["groups"] is None
            else {str(key): _accuracy_group(item) for key, item in value["groups"].items()}
        )
    return result


def _screening(value):
    result = {}
    if "residual" in value:
        result["residual"] = None if value["residual"] is None else _rejected(value["residual"])
    if "reported_sigma" in value:
        result["reportedSigma"] = None if value["reported_sigma"] is None else _accuracy(value["reported_sigma"])
    return result


def _weights(value):
    result = {}
    if "observation_count" in value:
        result["observationCount"] = (
            None if value["observation_count"] is None else plain_data(value["observation_count"])
        )
    if "full_weight_count" in value:
        result["fullWeightCount"] = (
            None if value["full_weight_count"] is None else plain_data(value["full_weight_count"])
        )
    if "downweighted_count" in value:
        result["downweightedCount"] = (
            None if value["downweighted_count"] is None else plain_data(value["downweighted_count"])
        )
    if "rejected_count" in value:
        result["rejectedCount"] = None if value["rejected_count"] is None else plain_data(value["rejected_count"])
    if "minimum" in value:
        result["minimum"] = None if value["minimum"] is None else plain_data(value["minimum"])
    if "p05" in value:
        result["p05"] = None if value["p05"] is None else plain_data(value["p05"])
    if "median" in value:
        result["median"] = None if value["median"] is None else plain_data(value["median"])
    if "p95" in value:
        result["p95"] = None if value["p95"] is None else plain_data(value["p95"])
    if "maximum" in value:
        result["maximum"] = None if value["maximum"] is None else plain_data(value["maximum"])
    return result


def _distribution(value):
    result = {}
    if "count" in value:
        result["count"] = None if value["count"] is None else plain_data(value["count"])
    if "rms" in value:
        result["rms"] = None if value["rms"] is None else plain_data(value["rms"])
    if "median" in value:
        result["median"] = None if value["median"] is None else plain_data(value["median"])
    if "mad" in value:
        result["mad"] = None if value["mad"] is None else plain_data(value["mad"])
    if "absolute_p50" in value:
        result["absoluteP50"] = None if value["absolute_p50"] is None else plain_data(value["absolute_p50"])
    if "absolute_p90" in value:
        result["absoluteP90"] = None if value["absolute_p90"] is None else plain_data(value["absolute_p90"])
    if "absolute_p95" in value:
        result["absoluteP95"] = None if value["absolute_p95"] is None else plain_data(value["absolute_p95"])
    if "absolute_p99" in value:
        result["absoluteP99"] = None if value["absolute_p99"] is None else plain_data(value["absolute_p99"])
    if "absolute_maximum" in value:
        result["absoluteMaximum"] = None if value["absolute_maximum"] is None else plain_data(value["absolute_maximum"])
    return result


def _distribution_m(value):
    result = {}
    if "count" in value:
        result["count"] = None if value["count"] is None else plain_data(value["count"])
    if "rms" in value:
        result["rms"] = None if value["rms"] is None else plain_data(value["rms"])
    if "median" in value:
        result["median"] = None if value["median"] is None else plain_data(value["median"])
    if "mad" in value:
        result["mad"] = None if value["mad"] is None else plain_data(value["mad"])
    if "absolute_p50" in value:
        result["absoluteP50"] = None if value["absolute_p50"] is None else plain_data(value["absolute_p50"])
    if "absolute_p90" in value:
        result["absoluteP90"] = None if value["absolute_p90"] is None else plain_data(value["absolute_p90"])
    if "absolute_p95" in value:
        result["absoluteP95"] = None if value["absolute_p95"] is None else plain_data(value["absolute_p95"])
    if "absolute_p99" in value:
        result["absoluteP99"] = None if value["absolute_p99"] is None else plain_data(value["absolute_p99"])
    if "absolute_maximum" in value:
        result["absoluteMaximum"] = None if value["absolute_maximum"] is None else plain_data(value["absolute_maximum"])
    return result


def _variance(value):
    result = {}
    if "active_count" in value:
        result["activeCount"] = None if value["active_count"] is None else plain_data(value["active_count"])
    if "observation_count" in value:
        result["observationCount"] = (
            None if value["observation_count"] is None else plain_data(value["observation_count"])
        )
    if "retained_observation_count" in value:
        result["retainedObservationCount"] = (
            None if value["retained_observation_count"] is None else plain_data(value["retained_observation_count"])
        )
    if "full_weight_count" in value:
        result["fullWeightCount"] = (
            None if value["full_weight_count"] is None else plain_data(value["full_weight_count"])
        )
    if "downweighted_count" in value:
        result["downweightedCount"] = (
            None if value["downweighted_count"] is None else plain_data(value["downweighted_count"])
        )
    if "rejected_count" in value:
        result["rejectedCount"] = None if value["rejected_count"] is None else plain_data(value["rejected_count"])
    if "effective_redundancy" in value:
        result["effectiveRedundancy"] = (
            None if value["effective_redundancy"] is None else plain_data(value["effective_redundancy"])
        )
    if "weighted_square_sum" in value:
        result["weightedSquareSum"] = (
            None if value["weighted_square_sum"] is None else plain_data(value["weighted_square_sum"])
        )
    if "sigma_factor_before" in value:
        result["sigmaFactorBefore"] = (
            None if value["sigma_factor_before"] is None else plain_data(value["sigma_factor_before"])
        )
    if "sigma_factor_after" in value:
        result["sigmaFactorAfter"] = (
            None if value["sigma_factor_after"] is None else plain_data(value["sigma_factor_after"])
        )
    if "variance_ratio" in value:
        result["varianceRatio"] = None if value["variance_ratio"] is None else plain_data(value["variance_ratio"])
    if "initial_sigma_factor" in value:
        result["initialSigmaFactor"] = (
            None if value["initial_sigma_factor"] is None else plain_data(value["initial_sigma_factor"])
        )
    if "final_sigma_factor" in value:
        result["finalSigmaFactor"] = (
            None if value["final_sigma_factor"] is None else plain_data(value["final_sigma_factor"])
        )
    if "variance_factor" in value:
        result["varianceFactor"] = None if value["variance_factor"] is None else plain_data(value["variance_factor"])
    if "standardized_rms" in value:
        result["standardizedRms"] = None if value["standardized_rms"] is None else plain_data(value["standardized_rms"])
    if "median_standardized_residual" in value:
        result["medianStandardizedResidual"] = (
            None if value["median_standardized_residual"] is None else plain_data(value["median_standardized_residual"])
        )
    if "mad_standardized_residual" in value:
        result["madStandardizedResidual"] = (
            None if value["mad_standardized_residual"] is None else plain_data(value["mad_standardized_residual"])
        )
    if "update_status" in value:
        result["updateStatus"] = None if value["update_status"] is None else plain_data(value["update_status"])
    if "component_id" in value:
        result["componentId"] = None if value["component_id"] is None else plain_data(value["component_id"])
    if "configured_start" in value:
        result["configuredStart"] = None if value["configured_start"] is None else plain_data(value["configured_start"])
    if "configured_end" in value:
        result["configuredEnd"] = None if value["configured_end"] is None else plain_data(value["configured_end"])
    if "actual_start_epoch_utc" in value:
        result["actualStartEpochUtc"] = (
            None if value["actual_start_epoch_utc"] is None else plain_data(value["actual_start_epoch_utc"])
        )
    if "actual_start_epoch_local" in value:
        result["actualStartEpochLocal"] = (
            None if value["actual_start_epoch_local"] is None else plain_data(value["actual_start_epoch_local"])
        )
    if "actual_end_epoch_utc" in value:
        result["actualEndEpochUtc"] = (
            None if value["actual_end_epoch_utc"] is None else plain_data(value["actual_end_epoch_utc"])
        )
    if "actual_end_epoch_local" in value:
        result["actualEndEpochLocal"] = (
            None if value["actual_end_epoch_local"] is None else plain_data(value["actual_end_epoch_local"])
        )
    if "proposed_sigma_factor_applied" in value:
        result["proposedSigmaFactorApplied"] = (
            None
            if value["proposed_sigma_factor_applied"] is None
            else plain_data(value["proposed_sigma_factor_applied"])
        )
    if "residual_rms_m" in value:
        result["residualRmsM"] = None if value["residual_rms_m"] is None else plain_data(value["residual_rms_m"])
    if "residual_wrms_m" in value:
        result["residualWrmsM"] = None if value["residual_wrms_m"] is None else plain_data(value["residual_wrms_m"])
    if "reported_sigma_screening" in value:
        result["reportedSigmaScreening"] = (
            None if value["reported_sigma_screening"] is None else _accuracy_group(value["reported_sigma_screening"])
        )
    if "residual_summary_m" in value:
        result["residualSummaryM"] = (
            None if value["residual_summary_m"] is None else _distribution(value["residual_summary_m"])
        )
    if "standardized_residual_summary" in value:
        result["standardizedResidualSummary"] = (
            None
            if value["standardized_residual_summary"] is None
            else _distribution_m(value["standardized_residual_summary"])
        )
    if "weight_factor_summary" in value:
        result["weightFactorSummary"] = (
            None if value["weight_factor_summary"] is None else _weights(value["weight_factor_summary"])
        )
    if "final_state_proposed_weight_factor_summary" in value:
        result["finalStateProposedWeightFactorSummary"] = (
            None
            if value["final_state_proposed_weight_factor_summary"] is None
            else _weights(value["final_state_proposed_weight_factor_summary"])
        )
    return result


def _sigma_iteration(value):
    result = {}
    if "iteration" in value:
        result["iteration"] = None if value["iteration"] is None else plain_data(value["iteration"])
    if "adjustment_iteration" in value:
        result["adjustmentIteration"] = (
            None if value["adjustment_iteration"] is None else plain_data(value["adjustment_iteration"])
        )
    if "sigma_weight_iteration" in value:
        result["sigmaWeightIteration"] = (
            None if value["sigma_weight_iteration"] is None else plain_data(value["sigma_weight_iteration"])
        )
    if "active_observation_count" in value:
        result["activeObservationCount"] = (
            None if value["active_observation_count"] is None else plain_data(value["active_observation_count"])
        )
    if "rejected_observation_count" in value:
        result["rejectedObservationCount"] = (
            None if value["rejected_observation_count"] is None else plain_data(value["rejected_observation_count"])
        )
    if "elapsed_seconds" in value:
        result["elapsedSeconds"] = None if value["elapsed_seconds"] is None else plain_data(value["elapsed_seconds"])
    if "maximum_sigma_factor_change" in value:
        result["maximumSigmaFactorChange"] = (
            None if value["maximum_sigma_factor_change"] is None else plain_data(value["maximum_sigma_factor_change"])
        )
    if "maximum_weight_factor_change" in value:
        result["maximumWeightFactorChange"] = (
            None if value["maximum_weight_factor_change"] is None else plain_data(value["maximum_weight_factor_change"])
        )
    if "total_frozen_redundancy" in value:
        result["totalFrozenRedundancy"] = (
            None if value["total_frozen_redundancy"] is None else plain_data(value["total_frozen_redundancy"])
        )
    if "expected_total_redundancy" in value:
        result["expectedTotalRedundancy"] = (
            None if value["expected_total_redundancy"] is None else plain_data(value["expected_total_redundancy"])
        )
    if "normal_matrix_condition" in value:
        result["normalMatrixCondition"] = (
            None if value["normal_matrix_condition"] is None else plain_data(value["normal_matrix_condition"])
        )
    if "candidate_wrms_m" in value:
        result["candidateWrmsM"] = None if value["candidate_wrms_m"] is None else plain_data(value["candidate_wrms_m"])
    if "maximum_candidate_parameter_update_m" in value:
        result["maximumCandidateParameterUpdateM"] = (
            None
            if value["maximum_candidate_parameter_update_m"] is None
            else plain_data(value["maximum_candidate_parameter_update_m"])
        )
    if "candidate_update_by_block_m" in value:
        result["candidateUpdateByBlockM"] = (
            None
            if value["candidate_update_by_block_m"] is None
            else {str(key): plain_data(item) for key, item in value["candidate_update_by_block_m"].items()}
        )
    if "sigma_factors" in value:
        result["sigmaFactors"] = (
            None
            if value["sigma_factors"] is None
            else {str(key): plain_data(item) for key, item in value["sigma_factors"].items()}
        )
    if "weight_factor_summary" in value:
        result["weightFactorSummary"] = (
            None if value["weight_factor_summary"] is None else _weights(value["weight_factor_summary"])
        )
    if "variance_components" in value:
        result["varianceComponents"] = (
            None
            if value["variance_components"] is None
            else {str(key): _variance(item) for key, item in value["variance_components"].items()}
        )
    return result


def _adjustment_iteration(value):
    result = {}
    if "iteration" in value:
        result["iteration"] = None if value["iteration"] is None else plain_data(value["iteration"])
    if "sigma_weight_iterations" in value:
        result["sigmaWeightIterations"] = (
            None if value["sigma_weight_iterations"] is None else plain_data(value["sigma_weight_iterations"])
        )
    if "equation_count" in value:
        result["equationCount"] = None if value["equation_count"] is None else plain_data(value["equation_count"])
    if "normal_matrix_rank" in value:
        result["normalMatrixRank"] = (
            None if value["normal_matrix_rank"] is None else plain_data(value["normal_matrix_rank"])
        )
    if "maximum_parameter_update_m" in value:
        result["maximumParameterUpdateM"] = (
            None if value["maximum_parameter_update_m"] is None else plain_data(value["maximum_parameter_update_m"])
        )
    if "wrms_m" in value:
        result["wrmsM"] = None if value["wrms_m"] is None else plain_data(value["wrms_m"])
    if "parameter_update_within_threshold" in value:
        result["parameterUpdateWithinThreshold"] = (
            None
            if value["parameter_update_within_threshold"] is None
            else plain_data(value["parameter_update_within_threshold"])
        )
    if "parameter_converged" in value:
        result["parameterConverged"] = (
            None if value["parameter_converged"] is None else plain_data(value["parameter_converged"])
        )
    if "convergence_threshold_by_parametrization_m" in value:
        result["convergenceThresholdByParametrizationM"] = (
            None
            if value["convergence_threshold_by_parametrization_m"] is None
            else {
                str(key): plain_data(item) for key, item in value["convergence_threshold_by_parametrization_m"].items()
            }
        )
    if "applied_update_by_block_m" in value:
        result["appliedUpdateByBlockM"] = (
            None
            if value["applied_update_by_block_m"] is None
            else {str(key): plain_data(item) for key, item in value["applied_update_by_block_m"].items()}
        )
    if "candidate_update_by_block_m" in value:
        result["candidateUpdateByBlockM"] = (
            None
            if value["candidate_update_by_block_m"] is None
            else {str(key): plain_data(item) for key, item in value["candidate_update_by_block_m"].items()}
        )
    if "candidate_parameter_corrections_m" in value:
        result["candidateParameterCorrectionsM"] = (
            None
            if value["candidate_parameter_corrections_m"] is None
            else {str(key): plain_data(item) for key, item in value["candidate_parameter_corrections_m"].items()}
        )
    if "normalized_parameter_update_by_block" in value:
        result["normalizedParameterUpdateByBlock"] = (
            None
            if value["normalized_parameter_update_by_block"] is None
            else {str(key): plain_data(item) for key, item in value["normalized_parameter_update_by_block"].items()}
        )
    if "sigma_factors_used_in_solve" in value:
        result["sigmaFactorsUsedInSolve"] = (
            None
            if value["sigma_factors_used_in_solve"] is None
            else {str(key): plain_data(item) for key, item in value["sigma_factors_used_in_solve"].items()}
        )
    if "sigma_factors_for_next_iteration" in value:
        result["sigmaFactorsForNextIteration"] = (
            None
            if value["sigma_factors_for_next_iteration"] is None
            else {str(key): plain_data(item) for key, item in value["sigma_factors_for_next_iteration"].items()}
        )
    if "normal_matrix_condition" in value:
        result["normalMatrixCondition"] = (
            None if value["normal_matrix_condition"] is None else plain_data(value["normal_matrix_condition"])
        )
    if "weight_factor_summary_used_in_solve" in value:
        result["weightFactorSummaryUsedInSolve"] = (
            None
            if value["weight_factor_summary_used_in_solve"] is None
            else _weights(value["weight_factor_summary_used_in_solve"])
        )
    if "weight_factor_summary_for_next_iteration" in value:
        result["weightFactorSummaryForNextIteration"] = (
            None
            if value["weight_factor_summary_for_next_iteration"] is None
            else _weights(value["weight_factor_summary_for_next_iteration"])
        )
    if "state_after_update" in value:
        result["stateAfterUpdate"] = (
            None
            if value["state_after_update"] is None
            else {
                str(key): (_station_state(item) if key == "stationRangeBias" else plain_data(item))
                for key, item in value["state_after_update"].items()
            }
        )
    return result


def _performance(value):
    result = {}
    if "cache_build" in value:
        result["cacheBuild"] = None if value["cache_build"] is None else plain_data(value["cache_build"])
    if "normal_solve" in value:
        result["normalSolve"] = None if value["normal_solve"] is None else plain_data(value["normal_solve"])
    if "redundancy" in value:
        result["redundancy"] = None if value["redundancy"] is None else plain_data(value["redundancy"])
    if "adjust_sigma0" in value:
        result["adjustSigma0"] = None if value["adjust_sigma0"] is None else plain_data(value["adjust_sigma0"])
    return result


def serialize_summary(value):
    result = {}
    if "converged" in value:
        result["converged"] = None if value["converged"] is None else plain_data(value["converged"])
    if "termination_reason" in value:
        result["terminationReason"] = (
            None if value["termination_reason"] is None else plain_data(value["termination_reason"])
        )
    if "source_observation_count" in value:
        result["sourceObservationCount"] = (
            None if value["source_observation_count"] is None else plain_data(value["source_observation_count"])
        )
    if "initial_light_time_converged_count" in value:
        result["initialLightTimeConvergedCount"] = (
            None
            if value["initial_light_time_converged_count"] is None
            else plain_data(value["initial_light_time_converged_count"])
        )
    if "gross_rejected_count" in value:
        result["grossRejectedCount"] = (
            None if value["gross_rejected_count"] is None else plain_data(value["gross_rejected_count"])
        )
    if "accuracy_rejected_count" in value:
        result["accuracyRejectedCount"] = (
            None if value["accuracy_rejected_count"] is None else plain_data(value["accuracy_rejected_count"])
        )
    if "retained_observation_count" in value:
        result["retainedObservationCount"] = (
            None if value["retained_observation_count"] is None else plain_data(value["retained_observation_count"])
        )
    if "last_solved_equation_count" in value:
        result["lastSolvedEquationCount"] = (
            None if value["last_solved_equation_count"] is None else plain_data(value["last_solved_equation_count"])
        )
    if "equation_evaluation_count" in value:
        result["equationEvaluationCount"] = (
            None if value["equation_evaluation_count"] is None else plain_data(value["equation_evaluation_count"])
        )
    if "adjustment_iteration_count" in value:
        result["adjustmentIterationCount"] = (
            None if value["adjustment_iteration_count"] is None else plain_data(value["adjustment_iteration_count"])
        )
    if "sigma_weight_iteration_count" in value:
        result["sigmaWeightIterationCount"] = (
            None if value["sigma_weight_iteration_count"] is None else plain_data(value["sigma_weight_iteration_count"])
        )
    if "last_normal_matrix_rank" in value:
        result["lastNormalMatrixRank"] = (
            None if value["last_normal_matrix_rank"] is None else plain_data(value["last_normal_matrix_rank"])
        )
    if "final_equation_count" in value:
        result["finalEquationCount"] = (
            None if value["final_equation_count"] is None else plain_data(value["final_equation_count"])
        )
    if "observation_count" in value:
        result["observationCount"] = (
            None if value["observation_count"] is None else plain_data(value["observation_count"])
        )
    if "parameter_count" in value:
        result["parameterCount"] = None if value["parameter_count"] is None else plain_data(value["parameter_count"])
    if "rank" in value:
        result["rank"] = None if value["rank"] is None else plain_data(value["rank"])
    if "last_normal_matrix_condition" in value:
        result["lastNormalMatrixCondition"] = (
            None if value["last_normal_matrix_condition"] is None else plain_data(value["last_normal_matrix_condition"])
        )
    if "condition_number" in value:
        result["conditionNumber"] = None if value["condition_number"] is None else plain_data(value["condition_number"])
    if "sigma0_post" in value:
        result["sigma0Post"] = None if value["sigma0_post"] is None else plain_data(value["sigma0_post"])
    if "parameter_uncertainty_sigma_multiplier" in value:
        result["parameterUncertaintySigmaMultiplier"] = (
            None
            if value["parameter_uncertainty_sigma_multiplier"] is None
            else plain_data(value["parameter_uncertainty_sigma_multiplier"])
        )
    if "performance_seconds" in value:
        result["performanceSeconds"] = (
            None if value["performance_seconds"] is None else _performance(value["performance_seconds"])
        )
    return result


def _parameter(value):
    result = {}
    if "name" in value:
        result["name"] = None if value["name"] is None else plain_data(value["name"])
    if "type" in value:
        result["type"] = None if value["type"] is None else plain_data(value["type"])
    if "maximum_correlated_parameter" in value:
        result["maximumCorrelatedParameter"] = (
            None if value["maximum_correlated_parameter"] is None else plain_data(value["maximum_correlated_parameter"])
        )
    if "unit" in value:
        result["unit"] = None if value["unit"] is None else plain_data(value["unit"])
    if "remaining_linearized_correction_m" in value:
        result["remainingLinearizedCorrectionM"] = (
            None
            if value["remaining_linearized_correction_m"] is None
            else plain_data(value["remaining_linearized_correction_m"])
        )
    if "cofactor_uncertainty_m" in value:
        result["cofactorUncertaintyM"] = (
            None if value["cofactor_uncertainty_m"] is None else plain_data(value["cofactor_uncertainty_m"])
        )
    if "formal_uncertainty_m" in value:
        result["formalUncertaintyM"] = (
            None if value["formal_uncertainty_m"] is None else plain_data(value["formal_uncertainty_m"])
        )
    if "maximum_absolute_correlation" in value:
        result["maximumAbsoluteCorrelation"] = (
            None if value["maximum_absolute_correlation"] is None else plain_data(value["maximum_absolute_correlation"])
        )
    if "reference_value" in value:
        result["referenceValue"] = None if value["reference_value"] is None else plain_data(value["reference_value"])
    if "applied_update" in value:
        result["appliedUpdate"] = None if value["applied_update"] is None else plain_data(value["applied_update"])
    if "final_value" in value:
        result["finalValue"] = None if value["final_value"] is None else plain_data(value["final_value"])
    return result


def _residuals(value):
    result = {}
    if "residual_m" in value:
        result["residualM"] = None if value["residual_m"] is None else _distribution(value["residual_m"])
    if "standardized_residual" in value:
        result["standardizedResidual"] = (
            None if value["standardized_residual"] is None else _distribution_m(value["standardized_residual"])
        )
    if "equivalent_weighted_rms_m" in value:
        result["equivalentWeightedRmsM"] = (
            None if value["equivalent_weighted_rms_m"] is None else plain_data(value["equivalent_weighted_rms_m"])
        )
    if "weight_factors" in value:
        result["weightFactors"] = None if value["weight_factors"] is None else _weights(value["weight_factors"])
    return result


def _observation(value):
    result = {}
    if "observation_id" in value:
        result["observationId"] = None if value["observation_id"] is None else plain_data(value["observation_id"])
    if "epoch_utc" in value:
        result["epochUtc"] = None if value["epoch_utc"] is None else plain_data(value["epoch_utc"])
    if "epoch_local" in value:
        result["epochLocal"] = None if value["epoch_local"] is None else plain_data(value["epoch_local"])
    if "station_id" in value:
        result["stationId"] = None if value["station_id"] is None else plain_data(value["station_id"])
    if "station" in value:
        result["station"] = None if value["station"] is None else plain_data(value["station"])
    if "variance_component_id" in value:
        result["varianceComponentId"] = (
            None if value["variance_component_id"] is None else plain_data(value["variance_component_id"])
        )
    if "reported_sigma_screening_status" in value:
        result["reportedSigmaScreeningStatus"] = (
            None
            if value["reported_sigma_screening_status"] is None
            else plain_data(value["reported_sigma_screening_status"])
        )
    if "reported_sigma_screening_reason" in value:
        result["reportedSigmaScreeningReason"] = (
            None
            if value["reported_sigma_screening_reason"] is None
            else plain_data(value["reported_sigma_screening_reason"])
        )
    if "applied_weight_status" in value:
        result["appliedWeightStatus"] = (
            None if value["applied_weight_status"] is None else plain_data(value["applied_weight_status"])
        )
    if "final_state_proposed_weight_status" in value:
        result["finalStateProposedWeightStatus"] = (
            None
            if value["final_state_proposed_weight_status"] is None
            else plain_data(value["final_state_proposed_weight_status"])
        )
    if "light_time_converged" in value:
        result["lightTimeConverged"] = (
            None if value["light_time_converged"] is None else plain_data(value["light_time_converged"])
        )
    if "troposphere_clamped" in value:
        result["troposphereClamped"] = (
            None if value["troposphere_clamped"] is None else plain_data(value["troposphere_clamped"])
        )
    if "proposed_weight_factor_applied" in value:
        result["proposedWeightFactorApplied"] = (
            None
            if value["proposed_weight_factor_applied"] is None
            else plain_data(value["proposed_weight_factor_applied"])
        )
    if "apriori_sigma_m" in value:
        result["aprioriSigmaM"] = None if value["apriori_sigma_m"] is None else plain_data(value["apriori_sigma_m"])
    if "minimum_valid_sigma_m" in value:
        result["minimumValidSigmaM"] = (
            None if value["minimum_valid_sigma_m"] is None else plain_data(value["minimum_valid_sigma_m"])
        )
    if "sigma0_m" in value:
        result["sigma0M"] = None if value["sigma0_m"] is None else plain_data(value["sigma0_m"])
    if "current_state_residual_m" in value:
        result["currentStateResidualM"] = (
            None if value["current_state_residual_m"] is None else plain_data(value["current_state_residual_m"])
        )
    if "linearized_postfit_residual_m" in value:
        result["linearizedPostfitResidualM"] = (
            None
            if value["linearized_postfit_residual_m"] is None
            else plain_data(value["linearized_postfit_residual_m"])
        )
    if "residual_sigma_m" in value:
        result["residualSigmaM"] = None if value["residual_sigma_m"] is None else plain_data(value["residual_sigma_m"])
    if "base_variance_m2" in value:
        result["baseVarianceM2"] = None if value["base_variance_m2"] is None else plain_data(value["base_variance_m2"])
    if "base_weight_per_m2" in value:
        result["baseWeightPerM2"] = (
            None if value["base_weight_per_m2"] is None else plain_data(value["base_weight_per_m2"])
        )
    if "equivalent_weight_per_m2" in value:
        result["equivalentWeightPerM2"] = (
            None if value["equivalent_weight_per_m2"] is None else plain_data(value["equivalent_weight_per_m2"])
        )
    if "sigma_factor" in value:
        result["sigmaFactor"] = None if value["sigma_factor"] is None else plain_data(value["sigma_factor"])
    if "leverage" in value:
        result["leverage"] = None if value["leverage"] is None else plain_data(value["leverage"])
    if "standardized_residual" in value:
        result["standardizedResidual"] = (
            None if value["standardized_residual"] is None else plain_data(value["standardized_residual"])
        )
    if "applied_weight_factor" in value:
        result["appliedWeightFactor"] = (
            None if value["applied_weight_factor"] is None else plain_data(value["applied_weight_factor"])
        )
    if "final_state_proposed_weight_factor" in value:
        result["finalStateProposedWeightFactor"] = (
            None
            if value["final_state_proposed_weight_factor"] is None
            else plain_data(value["final_state_proposed_weight_factor"])
        )
    if "matched_parameter_names" in value:
        result["matchedParameterNames"] = (
            None
            if value["matched_parameter_names"] is None
            else [plain_data(item) for item in value["matched_parameter_names"]]
        )
    return result


def serialize_state(value):
    return {
        str(key): (_station_state(item) if key == "stationRangeBias" else plain_data(item))
        for key, item in value.items()
    }


def serialize_adjustment_result(result):
    """Build a canonical report directly from one current solver result."""
    return plain_data(
        {
            "converged": result.converged,
            "publicationQuality": result.publication_quality(),
            "terminationReason": result.termination_reason,
            "settings": serialize_settings(result.settings),
            "equationEvaluations": [_evaluation(item) for item in result.equation_evaluations],
            "parameterNames": names_to_strings(result.parameter_names),
            "state": serialize_state(result.state),
            "observationScreening": {
                "residual": {
                    "rejectedCount": len(result.gross_rejected),
                    "rejectedObservations": {str(key): value for key, value in result.gross_rejected.items()},
                },
                "reportedSigma": _accuracy(result.accuracy_screening),
            },
            "sigmaFactors": result.sigma_factors,
            "weightFactors": {str(key): value for key, value in result.weight_factors.items()},
            "sigmaWeightIterations": [_sigma_iteration(asdict(item)) for item in result.sigma_weight_iterations],
            "adjustmentIterations": [_adjustment_iteration(item) for item in result.adjustment_iterations],
            "summary": serialize_summary(result.summary),
            "parameters": [_parameter(item) for item in result.parameters],
            "globalResiduals": _residuals(result.global_residuals),
            "varianceComponents": [_variance(item) for item in result.variance_components],
            "observations": [_observation(item) for item in result.observations],
        }
    )
