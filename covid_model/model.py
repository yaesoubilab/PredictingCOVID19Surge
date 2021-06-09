from SimPy.Parameters import Constant
from apace.CalibrationSupport import FeasibleConditions
from apace.Compartment import Compartment, ChanceNode, DeathCompartment
from apace.Control import InterventionAffectingContacts, ConditionBasedDecisionRule
from apace.Event import EpiIndepEvent, EpiDepEvent, PoissonEvent
from apace.FeaturesAndConditions import FeatureSurveillance, FeatureIntervention, \
    ConditionOnFeatures, FeatureEpidemicTime, ConditionOnConditions
from apace.TimeSeries import SumIncidence, SumPrevalence, SumCumulativeIncidence, RatioTimeSeries
from covid_model.parameters import COVIDParameters
from definitions import AgeGroupsProfiles


def build_covid_model(model):
    """ populates the provided model with a COVID model.
    :param model: an empty EpiModel to be populated
    """

    # store the model settings in a local variable for faster processing
    sets = model.settings

    # parameters of the COVID model
    params = COVIDParameters()

    indexer = AgeGroupsProfiles(n_age_groups=params.nAgeGroups, n_profiles=params.nProfiles)

    Ss = [None] * indexer.nAgeGroups
    Vs = [None] * indexer.nAgeGroups
    Es = [None] * indexer.length
    Is = [None] * indexer.length
    Hs = [None] * indexer.length
    ICUs = [None] * indexer.length
    Rs = [None] * indexer.length
    Ds = [None] * indexer.length
    ifs_hosp = [None] * indexer.length
    ifs_icu = [None] * indexer.length
    if_novel_strain = [None] * indexer.nAgeGroups
    # events
    importation = [None] * indexer.nAgeGroups
    infection_in_S = [None] * indexer.length
    leaving_Es = [None] * indexer.length
    leaving_Is = [None] * indexer.length
    leaving_Hs = [None] * indexer.length
    leaving_ICUs = [None] * indexer.length
    leaving_Rs = [None] * indexer.length
    deaths_in_ICU = [None] * indexer.length
    vaccination_in_S = [None] * indexer.nAgeGroups
    vaccination_in_R = [None] * indexer.length
    losing_vaccine_immunity = [None] * indexer.nAgeGroups

    # --------- model compartments ---------
    for a in range(indexer.nAgeGroups):
        str_a = indexer.get_str_age(age_group=a)
        Ss[a] = Compartment(name='Susceptible-'+str_a, size_par=params.sizeSByAge[a],
                            susceptibility_params=[Constant(value=1), Constant(value=1)])
        Vs[a] = Compartment(name='Vaccinated-'+str_a, num_of_pathogens=2)

        for p in range(indexer.nProfiles):

            str_a_p = indexer.get_str_age_profile(age_group=a, profile=p)
            i = indexer.get_row_index(age_group=a, profile=p)

            # infectivity
            infectivity_params = [Constant(value=0), Constant(value=0)]
            infectivity_params[p] = params.infectivity[p]

            # -------- compartments ----------
            Es[i] = Compartment(name='Exposed-'+str_a_p,
                                size_par=params.sizeEProfile0ByAge[a] if p == 0 else Constant(value=0),
                                infectivity_params=infectivity_params, if_empty_to_eradicate=True)
            Is[i] = Compartment(name='Infectious-'+str_a_p,
                                infectivity_params=infectivity_params, if_empty_to_eradicate=True)
            Hs[i] = Compartment(name='Hospitalized-'+str_a_p, num_of_pathogens=2, if_empty_to_eradicate=True)
            ICUs[i] = Compartment(name='ICU-'+str_a_p, num_of_pathogens=2, if_empty_to_eradicate=True)
            Rs[i] = Compartment(name='Recovered-'+str_a_p, num_of_pathogens=2)
            Ds[i] = DeathCompartment(name='Death-'+str_a_p)

            # --------- chance nodes ---------
            # chance node to decide if a hospitalized individual would need intensive care
            ifs_icu[i] = ChanceNode(name='If ICU-'+str_a_p,
                                    destination_compartments=[ICUs[i], Hs[i]],
                                    probability_params=params.probICUIfHosp[p])
            # chance node to decide if an infected individual would get hospitalized
            ifs_hosp[i] = ChanceNode(name='If hospitalized-'+str_a_p,
                                     destination_compartments=[ifs_icu[i], Rs[i]],
                                     probability_params=params.probHospByAge[a][p])

        # if an imported cases is infected with the novel strain
        dest_if_novel = indexer.get_row_index(age_group=a, profile=1)
        dest_if_current = indexer.get_row_index(age_group=a, profile=0)
        if_novel_strain[a] = ChanceNode(name='If infected with the novel strain-'+str_a,
                                        destination_compartments=[Es[dest_if_novel], Es[dest_if_current]],
                                        probability_params=params.probNovelStrain)

        # --------- model outputs to collect ---------
        # set up prevalence, incidence, and cumulative incidence to collect
        if sets.ifCollectTrajsOfCompartments:

            Ss[a].setup_history(collect_prev=True)
            Vs[a].setup_history(collect_prev=True)

            for p in range(indexer.nProfiles):
                i = indexer.get_row_index(age_group=a, profile=p)
                Es[i].setup_history(collect_prev=True)
                Is[i].setup_history(collect_prev=True)
                Hs[i].setup_history(collect_prev=True)
                ICUs[i].setup_history(collect_prev=True)
                Rs[i].setup_history(collect_prev=True)
                Ds[i].setup_history(collect_cum_incd=True)

        # --------- model events ---------
        for p in range(indexer.nProfiles):

            str_a_p = indexer.get_str_age_profile(age_group=a, profile=p)
            i = indexer.get_row_index(age_group=a, profile=p)

            infection_in_S[i] = EpiDepEvent(
                name='Infection in S-'+str_a_p, destination=Es[i], generating_pathogen=p)
            leaving_Es[i] = EpiIndepEvent(
                name='Leaving E-'+str_a_p, rate_param=params.ratesOfLeavingE[p], destination=Is[i])
            leaving_Is[i] = EpiIndepEvent(
                name='Leaving I-'+str_a_p, rate_param=params.ratesOfLeavingI[p], destination=ifs_hosp[i])
            leaving_Hs[i] = EpiIndepEvent(
                name='Leaving H-'+str_a_p, rate_param=params.ratesOfLeavingHosp[p], destination=Rs[i])
            leaving_ICUs[i] = EpiIndepEvent(
                name='Leaving ICU-'+str_a_p, rate_param=params.ratesOfLeavingICU[p], destination=Rs[i])
            leaving_Rs[i] = EpiIndepEvent(
                name='Leaving R-'+str_a_p, rate_param=params.ratesOfLeavingR[p], destination=Ss[a])
            deaths_in_ICU[i] = EpiIndepEvent(
                name='Death in ICU-'+str_a_p, rate_param=params.ratesOfDeathInICU[p], destination=Ds[i])
            vaccination_in_R[i] = EpiIndepEvent(
                name='Vaccinating R-'+str_a_p, rate_param=params.vaccRate, destination=Vs[a])

        importation[a] = PoissonEvent(
            name='Importation-'+str_a, destination=if_novel_strain[a], rate_param=params.importRateByAge[a])
        vaccination_in_S[a] = EpiIndepEvent(
            name='Vaccinating S-'+str_a, rate_param=params.vaccRate, destination=Vs[a])
        losing_vaccine_immunity[a] = EpiIndepEvent(
            name='Losing vaccine immunity-'+str_a, rate_param=params.rateOfLosingVacImmunity, destination=Ss[a])

        # --------- connections of events and compartments ---------
        # attached epidemic events to compartments
        i_inf_event = indexer.get_row_index(age_group=a, profile=0)
        Ss[a].add_events(events=[infection_in_S[i_inf_event], infection_in_S[i_inf_event+1],
                                 vaccination_in_S[a], importation[a]])
        Vs[a].add_event(event=losing_vaccine_immunity[a])

        for p in range(indexer.nProfiles):
            i = indexer.get_row_index(age_group=a, profile=p)
            Es[i].add_event(event=leaving_Es[i])
            Is[i].add_event(event=leaving_Is[i])
            Hs[i].add_event(event=leaving_Hs[i])
            ICUs[i].add_events(events=[leaving_ICUs[i], deaths_in_ICU[i]])
            Rs[i].add_events(events=[leaving_Rs[i], vaccination_in_R[i]])

    # --------- outcomes for projections ---------
    deaths = SumIncidence(name='Total deaths',
                          compartments=Ds,
                          if_surveyed=True,
                          collect_cumulative_after_warm_up='s')

    # --------- set up economic evaluation outcomes ---------
    deaths.setup_econ_outcome(par_health_per_new_member=Constant(1))

    # --------- sum time-series ------
    # population size
    compartments = Ss
    compartments.extend(Vs)
    compartments.extend(Es)
    compartments.extend(Is)
    compartments.extend(Hs)
    compartments.extend(ICUs)
    compartments.extend(Rs)
    compartments.extend(Ds)
    
    pop_size = SumPrevalence(name='Population size',
                             compartments=compartments)
    incidence = SumIncidence(name='Incidence', compartments=Is,
                             first_nonzero_obs_marks_start_of_epidemic=True)
    in_hosp = SumPrevalence(name='# in hospital', compartments=Hs, if_surveyed=True)
    in_icu = SumPrevalence(name='# in ICU', compartments=ICUs, if_surveyed=True)
    cum_vaccinated = SumCumulativeIncidence(name='Vaccinated', compartments=Vs, if_surveyed=True)

    # % cases with novel strain
    Is_current = []
    Is_novel = []
    for a in range(indexer.nAgeGroups):
        Is_current.append(Is[indexer.get_row_index(age_group=a, profile=0)])
        Is_novel.append(Is[indexer.get_row_index(age_group=a, profile=1)])
    incidence_a = SumIncidence(name='Incidence-0', compartments=Is_current, if_surveyed=True)
    incidence_b = SumIncidence(name='Incidence-1', compartments=Is_novel, if_surveyed=True)
    perc_cases_b = RatioTimeSeries(name='% of cases infected with novel strain',
                                   numerator_sum_time_series=incidence_b,
                                   denominator_sum_time_series=incidence,
                                   if_surveyed=True)

    # case fatality
    case_fatality = RatioTimeSeries(name='Case fatality',
                                    numerator_sum_time_series=deaths,
                                    denominator_sum_time_series=incidence,
                                    if_surveyed=True)

    # % population vaccinated
    perc_vaccinated = RatioTimeSeries(name='% of population vaccinated',
                                      numerator_sum_time_series=cum_vaccinated,
                                      denominator_sum_time_series=pop_size,
                                      if_surveyed=True)

    incd_by_age = []
    in_hosp_by_age = []
    in_icu_by_age = []
    cum_death_by_age = []
    for a in range(indexer.nAgeGroups):
        i_current = indexer.get_row_index(age_group=a, profile=0)
        i_novel = indexer.get_row_index(age_group=a, profile=1)

        # age-distribution of incidence

        # age-distribution of hospitalized patients

        # age-distributions of ICU patients

        # age-distribution of deaths


    # --------- feasibility conditions ---------
    # add feasible ranges of icu occupancy
    if sets.calcLikelihood:
        in_icu.add_feasible_conditions(feasible_conditions=FeasibleConditions(feasible_max=4 * 10.3,
                                                                              min_threshold_to_hit=4))



    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        settings=sets, params=params, in_icu=in_icu)

    # --------- populate the model ---------
    chance_nodes = []
    chance_nodes.extend(ifs_hosp)
    chance_nodes.extend(ifs_icu)
    chance_nodes.extend(if_novel_strain)

    model.populate(compartments=compartments,
                   parameters=params,
                   chance_nodes=chance_nodes,
                   list_of_sum_time_series=[pop_size, incidence_a, incidence_b, incidence, in_hosp, in_icu,
                                            deaths, cum_vaccinated],
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
