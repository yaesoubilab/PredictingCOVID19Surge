from SimPy.Parameters import Constant
from apace.CalibrationSupport import FeasibleConditions
from apace.Compartment import Compartment, ChanceNode, DeathCompartment
from apace.Control import InterventionAffectingContacts, ConditionBasedDecisionRule
from apace.Event import EpiIndepEvent, EpiDepEvent, PoissonEvent
from apace.FeaturesAndConditions import FeatureSurveillance, FeatureIntervention, \
    ConditionOnFeatures, FeatureEpidemicTime, ConditionOnConditions
from apace.TimeSeries import SumIncidence, SumPrevalence, RatioTimeSeries
from covid_model.parameters import COVIDParameters, AgeGroups


def build_covid_model(model):
    """ populates the provided model with a COVID model.
    :param model: an empty EpiModel to be populated
    """

    # store the model settings in a local variable for faster processing
    sets = model.settings

    # parameters of the COVID model
    params = COVIDParameters()

    n_profile = params.nProfiles
    n_age_groups = params.nAgeGroups

    Ss = [None] * n_age_groups
    V_Imns = [None] * n_age_groups
    V_Suss = [None] * n_age_groups

    Es = [None] * n_profile
    Is = [None] * n_profile
    Hs = [None] * n_profile
    ICUs = [None] * n_profile
    Rs = [None] * n_profile
    Ds = [None] * n_profile
    ifs_hosp = [None] * n_profile
    ifs_icu = [None] * n_profile
    infections_in_S = [None] * n_profile
    infections_in_V_Sus = [None] * n_profile
    leaving_Es = [None] * n_profile
    leaving_Is = [None] * n_profile
    leaving_Hs = [None] * n_profile
    leaving_ICUs = [None] * n_profile
    leaving_Rs = [None] * n_profile
    deaths_in_ICU = [None] * n_profile
    vaccinations_in_R = [None] * n_profile

    # --------- model compartments ---------
    for a in range(len(AgeGroups)):
        Ss[a] = Compartment(name='Susceptible-'+str(a), size_par=params.sizeS[a],
                            susceptibility_params=[Constant(value=1), Constant(value=1)])
        V_Imns[a] = Compartment(name='Vaccinated-Immune'+str(a), num_of_pathogens=2)
        V_Suss[a] = Compartment(name='Vaccinated-Susceptible'+str(a), num_of_pathogens=2)

    for i in range(n_profile):

        str_i = str(i)
        infectivity_params = [Constant(value=0), Constant(value=0)]
        infectivity_params[i] = params.infectivity[i]

        # -------- compartments ----------
        Es[i] = Compartment(name='Exposed-'+str_i,
                            size_par=params.sizeE[i],
                            infectivity_params=infectivity_params)
        Is[i] = Compartment(name='Infectious-'+str_i,
                            size_par=params.sizeI[i],
                            infectivity_params=infectivity_params)
        Hs[i] = Compartment(name='Hospitalized-'+str_i, num_of_pathogens=2, if_empty_to_eradicate=True)
        ICUs[i] = Compartment(name='ICU-'+str_i, num_of_pathogens=2, if_empty_to_eradicate=True)
        Rs[i] = Compartment(name='Recovered-'+str_i, num_of_pathogens=2)
        Ds[i] = DeathCompartment(name='Death-'+str_i)

        # --------- chance nodes ---------
        # chance node to decide if a hospitalized individual would need intensive care
        ifs_icu[i] = ChanceNode(name='If ICU-'+str_i,
                                destination_compartments=[ICUs[i], Hs[i]],
                                probability_params=params.probICUIfHosp[i])
        # chance node to decide if an infected individual would get hospitalized
        ifs_hosp[i] = ChanceNode(name='If hospitalized-'+str_i,
                                 destination_compartments=[ifs_icu[i], Rs[i]],
                                 probability_params=params.probHosp[i])

    # if an imported cases is infected with the novel strain
    if_novel_strain = ChanceNode(name='If infected with the novel strain',
                                 destination_compartments=[Es[1], Es[0]],
                                 probability_params=params.probNovelStrain)

    # --------- model outputs to collect ---------
    # set up prevalence, incidence, and cumulative incidence to collect
    if sets.ifCollectTrajsOfCompartments:

        S.setup_history(collect_prev=True)
        V_Imn.setup_history(collect_prev=True)
        V_Sus.setup_history(collect_prev=True)

        for i in range(n_profile):
            Es[i].setup_history(collect_prev=True)
            Is[i].setup_history(collect_prev=True, collect_incd=True)
            Hs[i].setup_history(collect_prev=True)
            ICUs[i].setup_history(collect_prev=True)
            Rs[i].setup_history(collect_prev=True)
            Ds[i].setup_history(collect_cum_incd=True)

    # --------- model events ---------
    for i in range(n_profile):
        str_i = str(i)

        infections_in_S[i] = EpiDepEvent(
            name='Infection in S-'+str_i, destination=Es[i], generating_pathogen=i)
        infections_in_V_Sus[i] = EpiDepEvent(
            name='Infection in V_Sus-'+str_i, destination=Es[i], generating_pathogen=i)
        leaving_Es[i] = EpiIndepEvent(
            name='Leaving E-'+str_i, rate_param=params.ratesOfLeavingE[i], destination=Is[i])
        leaving_Is[i] = EpiIndepEvent(
            name='Leaving I-'+str_i, rate_param=params.ratesOfLeavingI[i], destination=ifs_hosp[i])
        leaving_Hs[i] = EpiIndepEvent(
            name='Leaving H-'+str_i, rate_param=params.ratesOfLeavingHosp[i], destination=Rs[i])
        leaving_ICUs[i] = EpiIndepEvent(
            name='Leaving ICU-'+str_i, rate_param=params.ratesOfLeavingICU[i], destination=Rs[i])
        leaving_Rs[i] = EpiIndepEvent(
            name='Leaving R-'+str_i, rate_param=params.ratesOfLeavingR[i], destination=S)
        deaths_in_ICU[i] = EpiIndepEvent(
            name='Death in ICU-'+str_i, rate_param=params.ratesOfDeathInICU[i], destination=Ds[i])
        vaccinations_in_R[i] = EpiIndepEvent(
            name='Vaccinating R-'+str_i, rate_param=params.vaccRate, destination=V_Imn)

    importation = PoissonEvent(
        name='Importation', destination=if_novel_strain, rate_param=params.importRate)
    vaccinating_S = EpiIndepEvent(
        name='Vaccinating S', rate_param=params.vaccRate, destination=V_Imn)
    losing_vaccine_immunity = EpiIndepEvent(
        name='Losing vaccine immunity', rate_param=params.rateOfLosingVacImmunity, destination=V_Sus)

    # --------- connections of events and compartments ---------
    # attached epidemic events to compartments
    S.add_events(events=[infections_in_S[0], infections_in_S[1], vaccinating_S, importation])
    V_Sus.add_events(events=infections_in_V_Sus)
    V_Imn.add_event(event=losing_vaccine_immunity)

    for i in range(n_profile):
        Es[i].add_event(event=leaving_Es[i])
        Is[i].add_event(event=leaving_Is[i])
        Hs[i].add_event(event=leaving_Hs[i])
        ICUs[i].add_events(events=[leaving_ICUs[i], deaths_in_ICU[i]])
        Rs[i].add_events(events=[leaving_Rs[i], vaccinations_in_R[i]])

    # --------- projections ---------
    deaths = SumIncidence(name='Total deaths',
                          compartments=Ds,
                          if_surveyed=True,
                          collect_cumulative_after_warm_up='s')

    # --------- set up economic evaluation outcomes ---------
    deaths.setup_econ_outcome(par_health_per_new_member=Constant(1))

    # --------- sum time-series ------
    # population size
    compartments = [S, V_Imn, V_Sus,
                    Es[0], Is[0], Hs[0], ICUs[0], Rs[0], Ds[0],
                    Es[1], Is[1], Hs[1], ICUs[1], Rs[1], Ds[1]]
    pop_size = SumPrevalence(name='Population size',
                             compartments=compartments)

    # --------- surveillance ---------
    # setup surveillance to check the start of the epidemic
    incidence = SumIncidence(name='Incidence', compartments=Is,
                             first_nonzero_obs_marks_start_of_epidemic=True)
    incidence_a = SumIncidence(name='Incidence-0', compartments=[Is[0]], if_surveyed=True)
    incidence_b = SumIncidence(name='Incidence-1', compartments=[Is[1]], if_surveyed=True)
    in_hosp = SumPrevalence(name='# in hospital', compartments=Hs, if_surveyed=True)
    in_icu = SumPrevalence(name='# in ICU', compartments=ICUs, if_surveyed=True)
    vaccinated = SumPrevalence(name='Vaccinated', compartments=[V_Imn, V_Sus], if_surveyed=True)

    # add feasible ranges of icu occupancy
    if sets.calcLikelihood:
        in_icu.add_feasible_conditions(feasible_conditions=FeasibleConditions(feasible_max=4*10.3,
                                                                              min_threshold_to_hit=4))

    # case fatality
    case_fatality = RatioTimeSeries(name='Case fatality',
                                    numerator_sum_time_series=deaths,
                                    denominator_sum_time_series=incidence,
                                    if_surveyed=True)
    # % cases with novel strain
    perc_cases_b = RatioTimeSeries(name='% of cases infected with strain B',
                                   numerator_sum_time_series=incidence_b,
                                   denominator_sum_time_series=incidence,
                                   if_surveyed=True)
    # % population vaccinated
    perc_vaccinated = RatioTimeSeries(name='% of population vaccinated',
                                      numerator_sum_time_series=vaccinated,
                                      denominator_sum_time_series=pop_size,
                                      if_surveyed=True)

    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        settings=sets, params=params, in_icu=in_icu)

    # --------- populate the model ---------
    model.populate(compartments=compartments,
                   parameters=params,
                   chance_nodes=[ifs_hosp[0], ifs_hosp[1], ifs_icu[0], ifs_icu[1], if_novel_strain],
                   list_of_sum_time_series=[pop_size, incidence_a, incidence_b, incidence, in_hosp, in_icu,
                                            deaths, vaccinated],
                   list_of_ratio_time_series=[case_fatality, perc_cases_b, perc_vaccinated],
                   interventions=interventions,
                   features=features,
                   conditions=conditions
                   )


def get_interventions_features_conditions(settings, params, in_icu):

    # --------- set up modeling physical distancing ---------

    # features
    feature_on_surveyed_icu = None
    feature_on_epi_time = None
    feature_on_pd_y1 = None

    # conditions
    pass_y1 = None
    on_condition_during_y1 = None
    off_condition = None
    off_condition_during_y1 = None

    # interventions
    pd_year_1 = None

    # --------- features ---------
    # defined on surveyed icu
    feature_on_surveyed_icu = FeatureSurveillance(name='Surveyed number in ICU',
                                                  sum_time_series_with_surveillance=in_icu)
    # feature on time
    feature_on_epi_time = FeatureEpidemicTime(name='epidemic time')

    # if year 1 has passed
    pass_y1 = ConditionOnFeatures(
        name='if year 1.5 has passed',
        features=[feature_on_epi_time],
        signs=['ge'],
        thresholds=[1.5])

    # use of physical distancing during year 1
    if settings.ifPDInCalibrationPeriod:
        # ---------- intervention -------
        pd_year_1 = InterventionAffectingContacts(
            name='Physical distancing Y1',
            par_change_in_contact_matrix=params.matrixChangeInContactsY1)

        # --------- features ---------
        # feature defined on the intervention
        feature_on_pd_y1 = FeatureIntervention(name='Status of pd Y1',
                                               intervention=pd_year_1)
        # --------- conditions ---------
        on_condition_during_y1 = ConditionOnFeatures(
            name='turn on pd Y1',
            features=[feature_on_epi_time, feature_on_pd_y1, feature_on_surveyed_icu],
            signs=['l', 'e', 'ge'],
            thresholds=[1.5, 0, params.pdY1Thresholds[0]])
        off_condition = ConditionOnFeatures(
            name='turn off pd',
            features=[feature_on_pd_y1, feature_on_surveyed_icu],
            signs=['e', 'l'],
            thresholds=[1, params.pdY1Thresholds[1]])
        off_condition_during_y1 = ConditionOnConditions(
            name='turn off pd Y1',
            conditions=[pass_y1, off_condition],
            logic='or')

        # --------- decision rule ---------
        decision_rule = ConditionBasedDecisionRule(
            default_switch_value=0,
            condition_to_turn_on=on_condition_during_y1,
            condition_to_turn_off=off_condition_during_y1)
        pd_year_1.add_decision_rule(decision_rule=decision_rule)

    # make the list of features, conditions, and interventions
    features = [feature_on_surveyed_icu, feature_on_epi_time, feature_on_pd_y1]
    conditions = [pass_y1, on_condition_during_y1, off_condition, off_condition_during_y1]

    interventions = [pd_year_1]

    return interventions, features, conditions
