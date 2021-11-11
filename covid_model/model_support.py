from apace.CalibrationSupport import FeasibleConditions
from apace.Control import InterventionAffectingContacts, ConditionBasedDecisionRule, PredeterminedDecisionRule
from apace.FeaturesAndConditions import FeatureSurveillance, FeatureIntervention, \
    ConditionOnFeatures, FeatureEpidemicTime, ConditionOnConditions
from covid_model.data import MAX_HOSP_OCC_RATE, MIN_HOSP_OCC_RATE, MAX_HOSP_RATE_OVERALL, MIN_HOSP_RATE_OVERALL, MAX_PREV_IMMUNE_FROM_INF
from definitions import FEASIBILITY_PERIOD, SIM_DURATION


def get_interventions_features_conditions(params, hosp_occupancy_rate, mitigating_strategies_on):
    """
    :param params: model parameters
    :param hosp_occupancy_rate:
    :param mitigating_strategies_on:
    :return: (interventions, features, conditions)
    """

    # --------- set up modeling physical distancing ---------

    # features
    f_on_surveyed_hosp = None
    f_on_epi_time = None
    f_on_y1_intv = None
    f_on_y2_intv = None

    # conditions
    con_feasibility_period_passed = None
    con_on_intv_y1 = None
    con_off_intv_y1_ever = None
    con_off_intv_y1 = None
    con_on_intv_y2 = None
    con_off_intv_y2_ever = None
    con_off_intv_y2 = None

    # interventions
    y1_intv = None
    y2_intv = None

    # --------- features ---------
    # defined on surveyed in hospital
    f_on_surveyed_hosp = FeatureSurveillance(
        name='Surveyed number in ICU',
        ratio_time_series_with_surveillance=hosp_occupancy_rate)
    # feature on time
    f_on_epi_time = FeatureEpidemicTime(name='epidemic time')

    # if feasibility period has passed
    con_feasibility_period_passed = ConditionOnFeatures(
        name='if year {} has passed'.format(FEASIBILITY_PERIOD),
        features=[f_on_epi_time],
        signs=['ge'],
        thresholds=[FEASIBILITY_PERIOD])

    # use of physical distancing during FEASIBILITY_PERIOD
    # ---------- intervention -------
    y1_intv = InterventionAffectingContacts(
        name='Physical distancing during calibration period',
        par_perc_change_in_contact_matrix=params.matrixOfPercChangeInContactsY1)
    y2_intv = InterventionAffectingContacts(
        name='Physical distancing during fall/winter',
        par_perc_change_in_contact_matrix=params.matrixOfPercChangeInContactsY2)

    # --------- features ---------
    # feature defined on the intervention
    f_on_y1_intv = FeatureIntervention(name='Status of pd in year 1',
                                       intervention=y1_intv)
    f_on_y2_intv = FeatureIntervention(name='Status of pd in year 2',
                                       intervention=y2_intv)

    # --------- conditions ---------
    con_on_intv_y1 = ConditionOnFeatures(
        name='turn on pd y1',
        features=[f_on_epi_time, f_on_y1_intv, f_on_surveyed_hosp],
        signs=['l', 'e', 'ge'],
        thresholds=[FEASIBILITY_PERIOD, 0, params.y1Thresholds[0]])
    con_on_intv_y2 = ConditionOnFeatures(
        name='turn on pd y2',
        features=[f_on_epi_time, f_on_y2_intv, f_on_surveyed_hosp],
        signs=['ge', 'e', 'ge'],
        thresholds=[FEASIBILITY_PERIOD, 0, params.y2Thresholds[0]])

    con_off_intv_y1_ever = ConditionOnFeatures(
        name='turn off pd y1 (ever)',
        features=[f_on_y1_intv, f_on_surveyed_hosp],
        signs=['e', 'l'],
        thresholds=[1, params.y1Thresholds[1]])
    con_off_intv_y2_ever = ConditionOnFeatures(
        name='turn off pd y2 (ever)',
        features=[f_on_y2_intv, f_on_surveyed_hosp],
        signs=['e', 'l'],
        thresholds=[1, params.y2Thresholds[1]])

    con_off_intv_y1 = ConditionOnConditions(
        name='turn off pd y1',
        conditions=[con_feasibility_period_passed, con_off_intv_y1_ever],
        logic='or')
    con_off_intv_y2 = ConditionOnConditions(
        name='turn off pd y2',
        conditions=[con_off_intv_y2_ever],
        logic='or')

    # --------- decision rule ---------
    decision_rule_y1 = ConditionBasedDecisionRule(
        default_switch_value=0,
        condition_to_turn_on=con_on_intv_y1,
        condition_to_turn_off=con_off_intv_y1)
    y1_intv.add_decision_rule(decision_rule=decision_rule_y1)

    if mitigating_strategies_on:
        decision_rule_y2 = ConditionBasedDecisionRule(
            default_switch_value=0,
            condition_to_turn_on=con_on_intv_y2,
            condition_to_turn_off=con_off_intv_y2)
    else:
        decision_rule_y2 = PredeterminedDecisionRule(predetermined_switch_value=0)
    y2_intv.add_decision_rule(decision_rule=decision_rule_y2)

    # make the list of features, conditions, and interventions
    features = [f_on_surveyed_hosp, f_on_epi_time, f_on_y1_intv, f_on_y2_intv]
    conditions = [con_feasibility_period_passed,
                  con_on_intv_y1, con_off_intv_y1_ever, con_off_intv_y1,
                  con_on_intv_y2, con_off_intv_y2_ever, con_off_intv_y2]

    interventions = [y1_intv, y2_intv]

    return interventions, features, conditions


def add_calibration_info(settings,
                         age_groups_profiles,
                         hosp_occupancy_rate,
                         new_hosp_rate_by_age,
                         prev_immune_from_inf,
                         cum_hosp_rate_by_age,
                         cum_vaccine_rate_by_age,
                         perc_incd_delta):

    # feasible ranges of hospital occupancy rate
    hosp_occupancy_rate.add_feasible_conditions(
        feasible_conditions=FeasibleConditions(
            feasible_max=MAX_HOSP_OCC_RATE / 100000,
            min_threshold_to_hit=MIN_HOSP_OCC_RATE / 100000,
            period=[0, SIM_DURATION]))

    # feasible ranges of hospitalization rate
    new_hosp_rate_by_age[0].add_feasible_conditions(
        feasible_conditions=FeasibleConditions(
            feasible_max=MAX_HOSP_RATE_OVERALL / 100000,
            min_threshold_to_hit=MIN_HOSP_RATE_OVERALL / 100000,
            period=[0, FEASIBILITY_PERIOD]))
    # for a in range(age_groups_profiles.nAgeGroups):
    #     new_hosp_rate_by_age[a+1].add_feasible_conditions(
    #         feasible_conditions=FeasibleConditions(feasible_max=MAX_HOSP_RATE_BY_AGE[a] / 100000,
    #                                                min_threshold_to_hit=MIN_HOSP_RATE_BY_AGE[a] / 100000))

    # feasibility range of prevalence of population with immunity after infection
    prev_immune_from_inf.add_feasible_conditions(
        feasible_conditions=FeasibleConditions(
            feasible_max=MAX_PREV_IMMUNE_FROM_INF / 100,
            period=[0, FEASIBILITY_PERIOD]))

    prev_immune_from_inf.add_calibration_targets(
        ratios=settings.prevImmFromInfMean, variances=settings.prevImmFromInfVar)

    # calibration information for the overall hospitalization rate
    cum_hosp_rate_by_age[0].add_calibration_targets(
        ratios=settings.cumHospRateMean, variances=settings.cumHospRateVar)

    # calibration information for hospitalization rate by age
    for a in range(age_groups_profiles.nAgeGroups):
        cum_hosp_rate_by_age[a + 1].add_calibration_targets(
            ratios=settings.cumHospRateByAgeMean[a], survey_sizes=settings.cumHospRateByAgeN[a])

    # calibration information for the overall vaccination coverage
    cum_vaccine_rate_by_age[0].add_calibration_targets(
        ratios=settings.cumVaccRateMean, survey_sizes=settings.cumVaccRateN)

    # by age
    for a in range(age_groups_profiles.nAgeGroups):
        if a > 1:  # no age 0-4 and 5-12
            cum_vaccine_rate_by_age[a + 1].add_calibration_targets(
                ratios=settings.cumVaccRateByAgeMean[a], variances=settings.cumVaccRateByAgeVar[a])

    # calibration information for the percentage of infection associated with the novel variant
    # perc_incd_delta.add_calibration_targets(
    #     ratios=settings.percInfWithNovelMean, survey_sizes=settings.percInfWithNovelN,
    #     # variances=settings.percInfWithNovelVar
    # )
