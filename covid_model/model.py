from SimPy.Parameters import Constant
from apace.CalibrationSupport import FeasibleConditions
from apace.Control import InterventionAffectingContacts, ConditionBasedDecisionRule
from apace.FeaturesAndConditions import FeatureSurveillance, FeatureIntervention, \
    ConditionOnFeatures, FeatureEpidemicTime, ConditionOnConditions
from apace.ModelObjects import Compartment, ChanceNode, DeathCompartment, EpiIndepEvent, EpiDepEvent, PoissonEvent
from apace.TimeSeries import SumIncidence, SumPrevalence, SumCumulativeIncidence, RatioTimeSeries
from covid_model.data import MAX_HOSP_RATE_OVERALL, MIN_HOSP_RATE_OVERALL
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

    age_groups_profiles = AgeGroupsProfiles(n_age_groups=params.nAgeGroups, n_profiles=params.nProfiles)

    Ss = [None] * age_groups_profiles.nAgeGroups
    Vs = [None] * age_groups_profiles.nAgeGroups
    Es = [None] * age_groups_profiles.length
    Is = [None] * age_groups_profiles.length
    Hs = [None] * age_groups_profiles.length
    Rs = [None] * age_groups_profiles.length
    Ds = [None] * age_groups_profiles.length
    ifs_hosp = [None] * age_groups_profiles.length
    if_novel_strain = [None] * age_groups_profiles.nAgeGroups
    # events
    importation = [None] * age_groups_profiles.nAgeGroups
    infection_in_S = [None] * age_groups_profiles.length
    infection_in_V = [None] * age_groups_profiles.nAgeGroups
    leaving_Es = [None] * age_groups_profiles.length
    leaving_Is = [None] * age_groups_profiles.length
    leaving_Hs = [None] * age_groups_profiles.length
    leaving_Rs = [None] * age_groups_profiles.length
    deaths_in_hosp = [None] * age_groups_profiles.length
    vaccination_in_S = [None] * age_groups_profiles.nAgeGroups
    vaccination_in_R = [None] * age_groups_profiles.length
    losing_vaccine_immunity = [None] * age_groups_profiles.nAgeGroups

    # --------- model compartments ---------
    for a in range(age_groups_profiles.nAgeGroups):
        str_a = age_groups_profiles.get_str_age(age_group=a)
        Ss[a] = Compartment(name='Susceptible-'+str_a, size_par=params.sizeSByAge[a],
                            susceptibility_params=[Constant(value=1), Constant(value=1)],
                            row_index_contact_matrix=a)
        Vs[a] = Compartment(name='Vaccinated-'+str_a,
                            susceptibility_params=[Constant(value=0), params.suspVaccinatedAgainstNovel],
                            row_index_contact_matrix=a)

        for p in range(age_groups_profiles.nProfiles):

            str_a_p = age_groups_profiles.get_str_age_profile(age_group=a, profile=p)
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)

            # infectivity
            infectivity_params = [Constant(value=0), Constant(value=0), Constant(value=0)]
            infectivity_params[p] = params.infectivityByProfile[p]

            # -------- compartments ----------
            Es[i] = Compartment(name='Exposed-'+str_a_p,
                                num_of_pathogens=2, row_index_contact_matrix=a)
            Is[i] = Compartment(name='Infectious-'+str_a_p,
                                size_par=params.sizeIProfile0ByAge[a] if p == 0 else Constant(value=0),
                                infectivity_params=infectivity_params, if_empty_to_eradicate=True,
                                row_index_contact_matrix=a)
            Hs[i] = Compartment(name='Hospitalized-'+str_a_p,
                                num_of_pathogens=2, if_empty_to_eradicate=True,row_index_contact_matrix=a)
            Rs[i] = Compartment(name='Recovered-'+str_a_p,
                                num_of_pathogens=2, row_index_contact_matrix=a)
            Ds[i] = DeathCompartment(name='Death-'+str_a_p)

            # --------- chance nodes ---------
            # chance node to decide if an infected individual would get hospitalized
            ifs_hosp[i] = ChanceNode(name='If hospitalized-'+str_a_p,
                                     destination_compartments=[Hs[i], Rs[i]],
                                     probability_params=params.probHospByAgeAndProfile[a][p])

        # if an imported cases is infected with the novel strain
        dest_if_novel = age_groups_profiles.get_row_index(age_group=a, profile=1)
        dest_if_current = age_groups_profiles.get_row_index(age_group=a, profile=0)
        if_novel_strain[a] = ChanceNode(name='If infected with the novel strain-'+str_a,
                                        destination_compartments=[Es[dest_if_novel], Es[dest_if_current]],
                                        probability_params=params.probNovelStrain)

        # --------- model outputs to collect ---------
        # set up prevalence, incidence, and cumulative incidence to collect
        if sets.ifCollectTrajsOfCompartments:

            Ss[a].setup_history(collect_prev=True)
            Vs[a].setup_history(collect_prev=True)

            for p in range(age_groups_profiles.nProfiles):
                i = age_groups_profiles.get_row_index(age_group=a, profile=p)
                Es[i].setup_history(collect_prev=True)
                Is[i].setup_history(collect_prev=True)
                Hs[i].setup_history(collect_prev=True)
                Rs[i].setup_history(collect_prev=True)
                Ds[i].setup_history(collect_cum_incd=True)

        # --------- model events ---------
        for p in range(age_groups_profiles.nProfiles):

            str_a_p = age_groups_profiles.get_str_age_profile(age_group=a, profile=p)
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)

            if p in (0, 1):
                infection_in_S[i] = EpiDepEvent(
                    name='Infection in S-'+str_a_p, destination=Es[i], generating_pathogen=p)
            else:
                infection_in_V[a] = EpiDepEvent(
                    name='Infection in V-'+str_a_p, destination=Es[i], generating_pathogen=1)
            leaving_Es[i] = EpiIndepEvent(
                name='Leaving E-'+str_a_p, rate_param=params.ratesOfLeavingE[p], destination=Is[i])
            leaving_Is[i] = EpiIndepEvent(
                name='Leaving I-'+str_a_p, rate_param=params.ratesOfLeavingI[p], destination=ifs_hosp[i])
            leaving_Hs[i] = EpiIndepEvent(
                name='Leaving H-'+str_a_p, rate_param=params.ratesOfLeavingHosp[p], destination=Rs[i])
            leaving_Rs[i] = EpiIndepEvent(
                name='Leaving R-'+str_a_p, rate_param=params.ratesOfLeavingR[p], destination=Ss[a])
            deaths_in_hosp[i] = EpiIndepEvent(
                name='Death in H-'+str_a_p, rate_param=params.ratesOfDeathInHospByAge[a][p], destination=Ds[i])
            vaccination_in_R[i] = EpiIndepEvent(
                name='Vaccinating R-'+str_a_p, rate_param=params.vaccRateByAge[a], destination=Vs[a])

        importation[a] = PoissonEvent(
            name='Importation-'+str_a, destination=if_novel_strain[a], rate_param=params.importRateByAge[a])
        vaccination_in_S[a] = EpiIndepEvent(
            name='Vaccinating S-'+str_a, rate_param=params.vaccRateByAge[a], destination=Vs[a])
        losing_vaccine_immunity[a] = EpiIndepEvent(
            name='Losing vaccine immunity-'+str_a, rate_param=params.rateOfLosingVacImmunity, destination=Ss[a])

        # --------- connections of events and compartments ---------
        # attached epidemic events to compartments
        i_inf_event = age_groups_profiles.get_row_index(age_group=a, profile=0)
        Ss[a].add_events(events=[infection_in_S[i_inf_event],  # infection with dominant
                                 infection_in_S[i_inf_event+1],  # infection with novel
                                 vaccination_in_S[a],
                                 importation[a]])
        Vs[a].add_events(events=[losing_vaccine_immunity[a],
                                 infection_in_V[a]])

        for p in range(age_groups_profiles.nProfiles):
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)
            Es[i].add_event(event=leaving_Es[i])
            Is[i].add_event(event=leaving_Is[i])
            Hs[i].add_events(events=[leaving_Hs[i], deaths_in_hosp[i]])
            Rs[i].add_events(events=[leaving_Rs[i], vaccination_in_R[i]])

    # --------- sum time-series ------
    # population size
    compartments = Ss
    compartments.extend(Vs)
    compartments.extend(Es)
    compartments.extend(Is)
    compartments.extend(Hs)
    compartments.extend(Rs)
    compartments.extend(Ds)

    # lists to contain summation statistics
    pop_size_by_age = []
    incd_by_age = []
    new_hosp_by_age = []
    cum_incd_by_age = []
    cum_hosp_by_age = []
    cum_death_by_age = []
    cum_vaccine_by_age = []
    # rates
    incd_rate_by_age = []
    new_hosp_rate_by_age = []
    cum_hosp_rate_by_age = []
    cum_death_rate_by_age = []
    cum_vaccine_rate_by_age = []
    # age distributions
    age_dist_cum_incd = []
    age_dist_new_hosp = []
    age_dist_cum_death = []

    # population size 
    pop_size_by_age.append(SumPrevalence(
        name='Population size', compartments=compartments))
    # incidence 
    incd_by_age.append(SumIncidence(
        name='Incidence', compartments=Is, first_nonzero_obs_marks_start_of_epidemic=True, if_surveyed=True))
    # cumulative incidence
    cum_incd_by_age.append(SumCumulativeIncidence(
        name='Cumulative incidence', compartments=Is))
    # new hospitalization
    new_hosp_by_age.append(SumIncidence(
        name='New hospitalizations', compartments=Hs, if_surveyed=True))
    # cumulative hospitalization
    cum_hosp_by_age.append(SumCumulativeIncidence(
        name='Cumulative hospitalizations', compartments=Hs, if_surveyed=True))
    # cumulative death 
    cum_death_by_age.append(SumCumulativeIncidence(
        name='Cumulative death', compartments=Ds, if_surveyed=True))
    # cumulative vaccination 
    cum_vaccine_by_age.append(SumCumulativeIncidence(
        name='Cumulative vaccination', compartments=Vs, if_surveyed=True))

    # incidence rate
    incd_rate_by_age.append(RatioTimeSeries(name='Incidence rate',
                                            numerator_sum_time_series=incd_by_age[0],
                                            denominator_sum_time_series=pop_size_by_age[0],
                                            if_surveyed=True))
    # new hospitalization rate
    new_hosp_rate_by_age.append(RatioTimeSeries(name='New hospitalization rate',
                                                numerator_sum_time_series=new_hosp_by_age[0],
                                                denominator_sum_time_series=pop_size_by_age[0],
                                                if_surveyed=True))
    cum_hosp_rate_by_age.append(RatioTimeSeries(name='Cumulative hospitalization rate',
                                                numerator_sum_time_series=cum_hosp_by_age[0],
                                                denominator_sum_time_series=pop_size_by_age[0],
                                                if_surveyed=True))
    # cumulative death rate 
    cum_death_rate_by_age.append(RatioTimeSeries(name='Cumulative death rate',
                                                 numerator_sum_time_series=cum_death_by_age[0],
                                                 denominator_sum_time_series=pop_size_by_age[0],
                                                 if_surveyed=True))
    # cumulative vaccination rate
    cum_vaccine_rate_by_age.append(RatioTimeSeries(name='Cumulative vaccination rate',
                                                   numerator_sum_time_series=cum_vaccine_by_age[0],
                                                   denominator_sum_time_series=pop_size_by_age[0],
                                                   if_surveyed=True))
    
    # incidence and new hospitalization by profile (current, novel, vaccinated)
    incd_by_profile = []
    new_hosp_by_profile = []
    profile_dist_incd = []
    profile_dist_new_hosp = []
    for p in range(age_groups_profiles.nProfiles):
        # find Is and Hs in this profile
        Is_this_profile = []
        Hs_this_profile = []
        for a in range(age_groups_profiles.nAgeGroups):
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)
            Is_this_profile.append(Is[i])
            Hs_this_profile.append(Hs[i])

        str_profile = age_groups_profiles.get_str_profile(p)
        # incidence and hospitalization by profile
        incd_by_profile.append(SumIncidence(
            name='Incidence-'+str_profile, compartments=Is_this_profile))
        new_hosp_by_profile.append(SumIncidence(
            name='New hosp-'+str_profile, compartments=Hs_this_profile))

        # profile-distribution of incidence
        profile_dist_incd.append(RatioTimeSeries(name='% of incidence due to '+str_profile,
                                                 numerator_sum_time_series=incd_by_profile[-1],
                                                 denominator_sum_time_series=incd_by_age[0],
                                                 if_surveyed=True))
        # profile-distribution of new hospitalization
        profile_dist_new_hosp.append(RatioTimeSeries(name='% of new hospitalizations due to '+str_profile,
                                                     numerator_sum_time_series=new_hosp_by_profile[-1],
                                                     denominator_sum_time_series=new_hosp_by_age[0],
                                                     if_surveyed=True))
    
    # list to contain summation statistics
    for a in range(age_groups_profiles.nAgeGroups):
        str_a = age_groups_profiles.get_str_age(age_group=a)

        comparts_this_age = [Ss[a], Vs[a]]
        Is_this_age = []
        Hs_this_age = []
        Ds_this_age = []
        for p in range(age_groups_profiles.nProfiles):
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)
            comparts_this_age.extend([Es[i], Is[i], Hs[i], Rs[i]])
            Is_this_age.append(Is[i])
            Hs_this_age.append(Hs[i])
            Ds_this_age.append(Ds[i])

        # population size of this age group
        pop_size_by_age.append(SumPrevalence(
            name='Population-' + str_a, compartments=comparts_this_age))

        #  incidence
        incd_by_age.append(SumIncidence(
            name='Incidence-' + str_a, compartments=Is_this_age))
        # incidence rate
        incd_rate_by_age.append(RatioTimeSeries(
            name='Incidence rate-'+str_a,
            numerator_sum_time_series=incd_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1]))
        # cumulative incidence
        cum_incd_by_age.append(SumCumulativeIncidence(
            name='Cumulative incidence-' + str_a, compartments=Is_this_age))
        # age-distribution of cumulative incidence
        age_dist_cum_incd.append(RatioTimeSeries(
            name='Cumulative incidence-'+str_a+' (%)',
            numerator_sum_time_series=cum_incd_by_age[-1],
            denominator_sum_time_series=cum_incd_by_age[0]))

        # new hospitalizations
        new_hosp_by_age.append(SumIncidence(
            name='New hospitalizations-' + str_a, compartments=Hs_this_age))
        # rate of new hospitalizations
        new_hosp_rate_by_age.append(RatioTimeSeries(
            name='Hospitalization rate-'+str_a,
            numerator_sum_time_series=new_hosp_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1]))

        # cumulative hospitalizations
        cum_hosp_by_age.append(SumCumulativeIncidence(
            name='Cumulative hospitalizations-' + str_a, compartments=Hs_this_age))
        # rate of cumulative hospitalizations
        new_hosp_rate_by_age.append(RatioTimeSeries(
            name='Cumulative hospitalization rate-' + str_a,
            numerator_sum_time_series=cum_hosp_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution of cumulative hospitalizations
        age_dist_new_hosp.append(RatioTimeSeries(
            name='Cumulative hospitalizations-'+str_a+' (%)',
            numerator_sum_time_series=cum_hosp_by_age[-1],
            denominator_sum_time_series=cum_hosp_by_age[0]))

        # cumulative death
        cum_death_by_age.append(SumCumulativeIncidence(
            name='Cumulative death-' + str_a, compartments=Ds_this_age))
        # cumulative death rate
        cum_death_rate_by_age.append(RatioTimeSeries(
            name='Cumulative death rate-' + str_a,
            numerator_sum_time_series=cum_death_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution of cumulative death
        age_dist_cum_death.append(RatioTimeSeries(
            name='Cumulative death-'+str_a+' (%)',
            numerator_sum_time_series=cum_death_by_age[-1],
            denominator_sum_time_series=cum_death_by_age[0]))

        # cumulative vaccinations
        cum_vaccine_by_age.append(SumCumulativeIncidence(
            name='Cumulative vaccination-' + str_a, compartments=[Vs[a]]))
        # rate of cumulative vaccination
        cum_vaccine_rate_by_age.append(RatioTimeSeries(
            name='Cumulative vaccination rate-' + str_a,
            numerator_sum_time_series=cum_vaccine_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1]))

    # --------- calibration and feasibility conditions ---------
    if sets.calcLikelihood:
        # feasible ranges of hospitalization rate
        new_hosp_rate_by_age[0].add_feasible_conditions(
            feasible_conditions=FeasibleConditions(feasible_max=MAX_HOSP_RATE_OVERALL / 100000,
                                                   min_threshold_to_hit=MIN_HOSP_RATE_OVERALL / 100000))
        # for a in range(indexer.nAgeGroups):
        #     new_hosp_rate_by_age[a+1].add_feasible_conditions(
        #         feasible_conditions=FeasibleConditions(feasible_max=MAX_HOSP_RATE_BY_AGE[a] / 100000,
        #                                                min_threshold_to_hit=MIN_HOSP_RATE_BY_AGE[a] / 100000))

        # calibration information for the overall hospitalization rate
        cum_hosp_rate_by_age[0].add_calibration_targets(ratios=sets.cumHospRateMean,
                                                        survey_sizes=sets.cumHospRateN)

        # calibration information for the overall vaccination coverage
        cum_vaccine_rate_by_age[0].add_calibration_targets(ratios=sets.cumVaccRateMean,
                                                           survey_sizes=sets.cumVaccRateN)

    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        settings=sets, params=params, in_hosp_rate=new_hosp_rate_by_age[0])

    # --------- populate the model ---------
    # change nodes
    chance_nodes = []
    chance_nodes.extend(ifs_hosp)
    chance_nodes.extend(if_novel_strain)

    # summation-time series
    list_of_sum_time_series = []
    list_of_sum_time_series.extend(pop_size_by_age)
    list_of_sum_time_series.extend(incd_by_age)
    list_of_sum_time_series.extend(new_hosp_by_age)
    list_of_sum_time_series.extend(cum_incd_by_age)
    list_of_sum_time_series.extend(cum_hosp_by_age)
    list_of_sum_time_series.extend(cum_death_by_age)
    list_of_sum_time_series.extend(cum_vaccine_by_age)
    list_of_sum_time_series.extend(incd_by_profile)
    list_of_sum_time_series.extend(new_hosp_by_profile)

    # ratio time-series
    list_of_ratio_time_series = []
    list_of_ratio_time_series.extend(incd_rate_by_age)
    list_of_ratio_time_series.extend(new_hosp_rate_by_age)
    list_of_ratio_time_series.extend(cum_hosp_rate_by_age)
    list_of_ratio_time_series.extend(cum_death_rate_by_age)
    list_of_ratio_time_series.extend(cum_vaccine_rate_by_age)
    list_of_ratio_time_series.extend(profile_dist_incd)
    list_of_ratio_time_series.extend(profile_dist_new_hosp)
    list_of_ratio_time_series.extend(age_dist_cum_incd)
    list_of_ratio_time_series.extend(age_dist_new_hosp)
    list_of_ratio_time_series.extend(age_dist_cum_death)

    model.populate(compartments=compartments,
                   parameters=params,
                   param_base_contact_matrix=params.baseContactMatrix,
                   chance_nodes=chance_nodes,
                   list_of_sum_time_series=list_of_sum_time_series,
                   list_of_ratio_time_series=list_of_ratio_time_series,
                   interventions=interventions,
                   features=features,
                   conditions=conditions)


def get_interventions_features_conditions(settings, params, in_hosp_rate):

    # --------- set up modeling physical distancing ---------

    # features
    feature_on_surveyed_hosp = None
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
    # defined on surveyed in hospital
    feature_on_surveyed_hosp = FeatureSurveillance(name='Surveyed number in ICU',
                                                   ratio_time_series_with_surveillance=in_hosp_rate)
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
            par_perc_change_in_contact_matrix=params.matrixOfPercChangeInContactsY1)

        # --------- features ---------
        # feature defined on the intervention
        feature_on_pd_y1 = FeatureIntervention(name='Status of pd Y1',
                                               intervention=pd_year_1)
        # --------- conditions ---------
        on_condition_during_y1 = ConditionOnFeatures(
            name='turn on pd Y1',
            features=[feature_on_epi_time, feature_on_pd_y1, feature_on_surveyed_hosp],
            signs=['l', 'e', 'ge'],
            thresholds=[1.5, 0, params.pdY1Thresholds[0]])
        off_condition = ConditionOnFeatures(
            name='turn off pd',
            features=[feature_on_pd_y1, feature_on_surveyed_hosp],
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
    features = [feature_on_surveyed_hosp, feature_on_epi_time, feature_on_pd_y1]
    conditions = [pass_y1, on_condition_during_y1, off_condition, off_condition_during_y1]

    interventions = [pd_year_1]

    return interventions, features, conditions
