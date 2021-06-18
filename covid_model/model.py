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
    Rs = [None] * indexer.length
    Ds = [None] * indexer.length
    ifs_hosp = [None] * indexer.length
    if_novel_strain = [None] * indexer.nAgeGroups
    # events
    importation = [None] * indexer.nAgeGroups
    infection_in_S = [None] * indexer.length
    leaving_Es = [None] * indexer.length
    leaving_Is = [None] * indexer.length
    leaving_Hs = [None] * indexer.length
    leaving_Rs = [None] * indexer.length
    deaths_in_hosp = [None] * indexer.length
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
            Rs[i] = Compartment(name='Recovered-'+str_a_p, num_of_pathogens=2)
            Ds[i] = DeathCompartment(name='Death-'+str_a_p)

            # --------- chance nodes ---------
            # chance node to decide if an infected individual would get hospitalized
            ifs_hosp[i] = ChanceNode(name='If hospitalized-'+str_a_p,
                                     destination_compartments=[Hs[i], Rs[i]],
                                     probability_params=params.probHospByAgeAndProfile[a][p])

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
            leaving_Rs[i] = EpiIndepEvent(
                name='Leaving R-'+str_a_p, rate_param=params.ratesOfLeavingR[p], destination=Ss[a])
            deaths_in_hosp[i] = EpiIndepEvent(
                name='Death in H-'+str_a_p, rate_param=params.ratesOfDeathInHosp[p], destination=Ds[i])
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
    in_hosp_by_age = []
    cum_death_by_age = []
    cum_vaccine_by_age = []

    incd_rate_by_age = []
    hosp_rate_by_age = []
    cum_death_rate_by_age = []
    cum_vaccine_rate_by_age = []
    
    # population size 
    pop_size_by_age.append(SumPrevalence(name='Population size',
                                         compartments=compartments))
    # incidence 
    incd_by_age.append(SumIncidence(name='Incidence', compartments=Is,
                                    first_nonzero_obs_marks_start_of_epidemic=True, if_surveyed=True))
    # in hospital 
    in_hosp_by_age.append(SumPrevalence(name='# in hospital', compartments=Hs, if_surveyed=True))
    # cumulative death 
    cum_death_by_age.append(SumCumulativeIncidence(name='Cumulative death', compartments=Ds, if_surveyed=True))
    # cumulative vaccination 
    cum_vaccine_by_age.append(SumCumulativeIncidence(name='Cumulative vaccination', compartments=Vs, if_surveyed=True))

    # incidence rate
    incd_rate_by_age.append(RatioTimeSeries(name='Incidence rate',
                                            numerator_sum_time_series=incd_by_age[-1],
                                            denominator_sum_time_series=pop_size_by_age[-1],
                                            if_surveyed=True))
    # hospitalization rate
    hosp_rate_by_age.append(RatioTimeSeries(name='Hospitalization rate',
                                            numerator_sum_time_series=in_hosp_by_age[-1],
                                            denominator_sum_time_series=pop_size_by_age[-1],
                                            if_surveyed=True))
    # cumulative death rate 
    cum_death_rate_by_age.append(RatioTimeSeries(name='Cumulative death rate',
                                                 numerator_sum_time_series=cum_death_by_age[-1],
                                                 denominator_sum_time_series=pop_size_by_age[-1],
                                                 if_surveyed=True))
    # cumulative vaccination rate
    cum_vaccine_rate_by_age.append(RatioTimeSeries(name='Cumulative vaccination rate',
                                                   numerator_sum_time_series=cum_vaccine_by_age[-1],
                                                   denominator_sum_time_series=pop_size_by_age[-1],
                                                   if_surveyed=True))
    
    # new hospitalization by profile (current, novel, vaccinated)
    new_hosp_by_profile = []
    profile_dist_new_hosp = []
    for p in range(indexer.nProfiles):
        # find Hs in this profile
        Hs_this_profile = []
        for a in range(indexer.nAgeGroups):
            i = indexer.get_row_index(age_group=a, profile=p)
            Hs_this_profile.append(Hs[i])

        str_profile = indexer.get_str_profile(p)
        # hospitalization by profile
        new_hosp_by_profile.append(SumIncidence(name='New hosp-'+str_profile,
                                                compartments=Hs_this_profile))
        # profile-distribution of new hospitalization
        profile_dist_new_hosp.append(RatioTimeSeries(name='% of new hospitalizations due to '+str_profile,
                                                     numerator_sum_time_series=new_hosp_by_profile[-1],
                                                     denominator_sum_time_series=in_hosp_by_age[0],
                                                     if_surveyed=True))
    
    # list to contain summation statistics
    age_dist_incd = []
    age_dist_in_hosp = []
    age_dist_cum_death = []
    age_dist_cum_vaccine = []
    for a in range(indexer.nAgeGroups):
        str_a = indexer.get_str_age(age_group=a)

        comparts_this_age = [Ss[a], Vs[a]]
        Is_this_age = []
        Hs_this_age = []
        Ds_this_age = []
        for p in range(indexer.nProfiles):
            i = indexer.get_row_index(age_group=a, profile=p)
            comparts_this_age.extend([Es[i], Is[i], Hs[i], Rs[i]])
            Is_this_age.append(Is[i])
            Hs_this_age.append(Hs[i])
            Ds_this_age.append(Ds[i])

        # population size of this age group
        pop_size_by_age.append(SumPrevalence(name='Population-' + str_a,
                                             compartments=comparts_this_age))

        #  incidence
        incd_by_age.append(SumIncidence(name='Incidence-' + str_a,
                                        compartments=Is_this_age))
        # rate
        incd_rate_by_age.append(RatioTimeSeries(name='Incidence rate-'+str_a,
                                                numerator_sum_time_series=incd_by_age[-1],
                                                denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution
        age_dist_incd.append(RatioTimeSeries(name='Incidence-'+str_a+' (%)',
                                             numerator_sum_time_series=incd_by_age[-1],
                                             denominator_sum_time_series=incd_by_age[0]))

        # hospitalization
        in_hosp_by_age.append(SumPrevalence(name='Hospitalized-' + str_a,
                                            compartments=Hs_this_age))
        # rate
        hosp_rate_by_age.append(RatioTimeSeries(name='Hospitalization rate-'+str_a,
                                                numerator_sum_time_series=in_hosp_by_age[-1],
                                                denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution
        age_dist_in_hosp.append(RatioTimeSeries(name='Hospitalized-'+str_a+' (%)',
                                                numerator_sum_time_series=in_hosp_by_age[-1],
                                                denominator_sum_time_series=in_hosp_by_age[0]))
        # cumulative death
        cum_death_by_age.append(SumCumulativeIncidence(name='Cumulative death-' + str_a,
                                                       compartments=Ds_this_age))
        # rate
        cum_death_rate_by_age.append(RatioTimeSeries(name='Cumulative death rate-' + str_a,
                                                     numerator_sum_time_series=cum_death_by_age[-1],
                                                     denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution
        age_dist_cum_death.append(RatioTimeSeries(name='Cumulative death-'+str_a+' (%)',
                                                  numerator_sum_time_series=cum_death_by_age[-1],
                                                  denominator_sum_time_series=cum_death_by_age[0]))

        # cumulative vaccinations
        cum_vaccine_by_age.append(SumCumulativeIncidence(name='Cumulative vaccination-' + str_a,
                                                         compartments=[Vs[-1]]))
        # rate
        cum_vaccine_rate_by_age.append(RatioTimeSeries(name='Cumulative vaccination rate-' + str_a,
                                                       numerator_sum_time_series=cum_vaccine_by_age[-1],
                                                       denominator_sum_time_series=pop_size_by_age[-1]))
        # age-distribution
        age_dist_cum_vaccine.append(RatioTimeSeries(name='Cumulative vaccination-'+str_a+' (%)',
                                                    numerator_sum_time_series=cum_vaccine_by_age[-1],
                                                    denominator_sum_time_series=cum_vaccine_by_age[0]))

    # --------- feasibility conditions ---------
    # add feasible ranges of icu occupancy
    if sets.calcLikelihood:
        hosp_rate_by_age[0].add_feasible_conditions(
            feasible_conditions=FeasibleConditions(feasible_max=20*4 * 10.3/100000,
                                                   min_threshold_to_hit=20/100000))

    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        settings=sets, params=params, in_hosp_rate=hosp_rate_by_age[0])

    # --------- populate the model ---------
    # change nodes
    chance_nodes = []
    chance_nodes.extend(ifs_hosp)
    chance_nodes.extend(if_novel_strain)

    # summation-time series
    list_of_sum_time_series = []
    list_of_sum_time_series.extend(pop_size_by_age)
    list_of_sum_time_series.extend(incd_by_age)
    list_of_sum_time_series.extend(in_hosp_by_age)
    list_of_sum_time_series.extend(cum_death_by_age)
    list_of_sum_time_series.extend(cum_vaccine_by_age)
    list_of_sum_time_series.extend(new_hosp_by_profile)

    # ratio time-series
    list_of_ratio_time_series = []
    list_of_ratio_time_series.extend(incd_rate_by_age)
    list_of_ratio_time_series.extend(hosp_rate_by_age)
    list_of_ratio_time_series.extend(cum_death_rate_by_age)
    list_of_ratio_time_series.extend(cum_vaccine_rate_by_age)
    list_of_ratio_time_series.extend(profile_dist_new_hosp)
    list_of_ratio_time_series.extend(age_dist_incd)
    list_of_ratio_time_series.extend(age_dist_in_hosp)
    list_of_ratio_time_series.extend(age_dist_cum_death)
    list_of_ratio_time_series.extend(age_dist_cum_vaccine)

    model.populate(compartments=compartments,
                   parameters=params,
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
            par_change_in_contact_matrix=params.matrixChangeInContactsY1)

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
