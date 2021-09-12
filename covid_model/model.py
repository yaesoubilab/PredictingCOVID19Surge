from SimPy.Parameters import Constant
from apace.CalibrationSupport import FeasibleConditions
from apace.Control import InterventionAffectingContacts, ConditionBasedDecisionRule
from apace.FeaturesAndConditions import FeatureSurveillance, FeatureIntervention, \
    ConditionOnFeatures, FeatureEpidemicTime, ConditionOnConditions
from apace.ModelObjects import Compartment, ChanceNode, Counter, \
    DeathCompartment, EpiIndepEvent, EpiDepEvent, PoissonEvent
from apace.TimeSeries import SumIncidence, SumPrevalence, SumCumulativeIncidence, RatioTimeSeries
from covid_model.data import MAX_HOSP_RATE_OVERALL, MIN_HOSP_RATE_OVERALL, MAX_PREV_IMMUNE_FROM_INF
from covid_model.parameters import COVIDParameters
from definitions import AgeGroupsProfiles, Profiles, FEASIBILITY_PERIOD


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
    ifs_novel_strain = [None] * age_groups_profiles.nAgeGroups
    counting_vacc_in_S = [None] * age_groups_profiles.nAgeGroups
    counting_vacc_in_R_dom = [None] * age_groups_profiles.nAgeGroups
    counting_vacc_in_R_nov = [None] * age_groups_profiles.nAgeGroups

    # events
    importation = [None] * age_groups_profiles.nAgeGroups
    infection_in_S = [None] * age_groups_profiles.length
    infection_in_V = [None] * age_groups_profiles.length
    leaving_Es = [None] * age_groups_profiles.length
    leaving_Is = [None] * age_groups_profiles.length
    leaving_Hs = [None] * age_groups_profiles.length
    leaving_Rs = [None] * age_groups_profiles.length
    deaths_in_hosp = [None] * age_groups_profiles.length
    vaccination_in_S = [None] * age_groups_profiles.nAgeGroups
    vaccination_in_R_dom = [None] * age_groups_profiles.nAgeGroups
    vaccination_in_R_nov = [None] * age_groups_profiles.nAgeGroups
    losing_vaccine_immunity = [None] * age_groups_profiles.length

    # --------- model compartments ---------
    for a in range(age_groups_profiles.nAgeGroups):
        str_a = age_groups_profiles.get_str_age(age_group=a)
        Ss[a] = Compartment(name='Susceptible-'+str_a, size_par=params.sizeSByAge[a],
                            susceptibility_params=[Constant(value=1), Constant(value=1)],
                            row_index_contact_matrix=a)
        Vs[a] = Compartment(name='Vaccinated-'+str_a,
                            susceptibility_params=params.suspVaccinated,
                            row_index_contact_matrix=a)

        for p in range(age_groups_profiles.nProfiles):

            str_a_p = age_groups_profiles.get_str_age_profile(age_group=a, profile=p)
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)

            # infectivity
            infectivity_params = [Constant(0), Constant(0)]
            infectivity_params[p % 2] = params.infectivityByProfile[p]

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

        # count vaccinations among recovered after infection with dominant or novel variants
        for p in (Profiles.DOM_UNVAC.value, Profiles.NOV_UNVAC.value):
            str_a_p = age_groups_profiles.get_str_age_profile(age_group=a, profile=p)

            if p == Profiles.DOM_UNVAC.value:
                dest_after_vacc_in_recovered = age_groups_profiles.get_row_index(age_group=a,
                                                                                 profile=Profiles.DOM_VAC.value)
                counting_vacc_in_R_dom[a] = Counter(name='Vaccination in R-' + str_a_p,
                                                    destination_compartment=Rs[dest_after_vacc_in_recovered])
            elif p == Profiles.NOV_UNVAC.value:
                dest_after_vacc_in_recovered = age_groups_profiles.get_row_index(age_group=a,
                                                                                 profile=Profiles.NOV_VAC.value)
                counting_vacc_in_R_nov[a] = Counter(name='Vaccination in R-' + str_a_p,
                                                    destination_compartment=Rs[dest_after_vacc_in_recovered])

        # if an imported cases is infected with the novel strain
        dest_if_novel = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.NOV_UNVAC.value)
        dest_if_dominant = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.DOM_UNVAC.value)
        ifs_novel_strain[a] = ChanceNode(name='If infected with the novel strain-'+str_a,
                                         destination_compartments=[Es[dest_if_novel], Es[dest_if_dominant]],
                                         probability_params=params.probNovelStrain)

        # count vaccinations among susceptibles
        counting_vacc_in_S[a] = Counter(name='Vaccination in S-' + str_a,
                                        destination_compartment=Vs[a])

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

            pathogen = p % 2
            if p in (Profiles.DOM_UNVAC.value, Profiles.NOV_UNVAC.value):
                infection_in_S[i] = EpiDepEvent(
                    name='Infection in S-'+str_a_p, destination=Es[i], generating_pathogen=pathogen)
            else:
                infection_in_V[i] = EpiDepEvent(
                    name='Infection in V-'+str_a_p, destination=Es[i], generating_pathogen=pathogen)
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

        importation[a] = PoissonEvent(
            name='Importation-'+str_a, destination=ifs_novel_strain[a], rate_param=params.importRateByAge[a])
        vaccination_in_S[a] = EpiIndepEvent(
            name='Vaccinating S-'+str_a, rate_param=params.vaccRateByAge[a], destination=counting_vacc_in_S[a])
        vaccination_in_R_dom[a] = EpiIndepEvent(
            name='Vaccinating R-dominant-' + str_a, rate_param=params.vaccRateByAge[a],
            destination=counting_vacc_in_R_dom[a])
        vaccination_in_R_nov[a] = EpiIndepEvent(
            name='Vaccinating R-novel-' + str_a, rate_param=params.vaccRateByAge[a],
            destination=counting_vacc_in_R_nov[a])
        losing_vaccine_immunity[a] = EpiIndepEvent(
            name='Losing vaccine immunity-'+str_a, rate_param=params.rateOfLosingVacImmunity, destination=Ss[a])

        # --------- connections of events and compartments ---------
        # attached epidemic events to compartments
        i_inf_with_dominant_event = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.DOM_UNVAC.value)
        i_inf_with_novel_event = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.NOV_UNVAC.value)
        Ss[a].add_events(events=[infection_in_S[i_inf_with_dominant_event],  # infection with dominant
                                 infection_in_S[i_inf_with_novel_event],  # infection with novel
                                 vaccination_in_S[a],
                                 importation[a]])
        i_inf_with_dominant_event = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.DOM_VAC.value)
        i_inf_with_novel_event = age_groups_profiles.get_row_index(age_group=a, profile=Profiles.NOV_VAC.value)
        Vs[a].add_events(events=[infection_in_V[i_inf_with_dominant_event],  # infection with dominant
                                 infection_in_V[i_inf_with_novel_event],  # infection with novel
                                 losing_vaccine_immunity[a],
                                 ])

        for p in range(age_groups_profiles.nProfiles):
            i = age_groups_profiles.get_row_index(age_group=a, profile=p)
            Es[i].add_event(event=leaving_Es[i])
            Is[i].add_event(event=leaving_Is[i])
            Hs[i].add_events(events=[leaving_Hs[i], deaths_in_hosp[i]])
            if p in (Profiles.DOM_UNVAC.value, Profiles.NOV_UNVAC.value):
                Rs[i].add_events(events=[leaving_Rs[i], vaccination_in_R_dom[a], vaccination_in_R_nov[a]])
            else:
                Rs[i].add_event(event=leaving_Rs[i])

    # --------- sum time-series ------
    # population size
    compartments = Ss + Vs + Es + Is + Hs + Rs + Ds

    # lists to contain summation statistics
    all_vaccinations = counting_vacc_in_S + counting_vacc_in_R_dom + counting_vacc_in_R_nov
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

    pop_size_by_age.append(SumPrevalence(
        name='Population size', compartments=compartments))
    num_susp = SumPrevalence(
        name='Individuals susceptible', compartments=Ss)
    num_immune_from_inf = SumPrevalence(
        name='Individuals wth immunity from infection', compartments=Rs)
    incd_by_age.append(SumIncidence(
        name='Incidence', compartments=Is, first_nonzero_obs_marks_start_of_epidemic=True, if_surveyed=True))
    cum_incd_by_age.append(SumCumulativeIncidence(
        name='Cumulative incidence', compartments=Is))
    new_hosp_by_age.append(SumIncidence(
        name='New hospitalizations', compartments=Hs, if_surveyed=True))
    cum_hosp_by_age.append(SumCumulativeIncidence(
        name='Cumulative hospitalizations', compartments=Hs, if_surveyed=True))
    cum_death_by_age.append(SumCumulativeIncidence(
        name='Cumulative death', compartments=Ds, if_surveyed=True))
    cum_vaccine_by_age.append(SumCumulativeIncidence(
        name='Cumulative vaccination', compartments=all_vaccinations, if_surveyed=True))

    prev_susp = RatioTimeSeries(name='Prevalence susceptible',
                                numerator_sum_time_series=num_susp,
                                denominator_sum_time_series=pop_size_by_age[0],
                                if_surveyed=True)
    prev_immune_from_inf = RatioTimeSeries(name='Prevalence with immunity from infection',
                                           numerator_sum_time_series=num_immune_from_inf,
                                           denominator_sum_time_series=pop_size_by_age[0],
                                           if_surveyed=True)
    incd_rate_by_age.append(RatioTimeSeries(name='Incidence rate',
                                            numerator_sum_time_series=incd_by_age[0],
                                            denominator_sum_time_series=pop_size_by_age[0],
                                            if_surveyed=True))
    new_hosp_rate_by_age.append(RatioTimeSeries(name='New hospitalization rate',
                                                numerator_sum_time_series=new_hosp_by_age[0],
                                                denominator_sum_time_series=pop_size_by_age[0],
                                                if_surveyed=True))
    cum_hosp_rate_by_age.append(RatioTimeSeries(name='Cumulative hospitalization rate',
                                                numerator_sum_time_series=cum_hosp_by_age[0],
                                                denominator_sum_time_series=pop_size_by_age[0],
                                                if_surveyed=True))
    cum_death_rate_by_age.append(RatioTimeSeries(name='Cumulative death rate',
                                                 numerator_sum_time_series=cum_death_by_age[0],
                                                 denominator_sum_time_series=pop_size_by_age[0],
                                                 if_surveyed=True))
    cum_vaccine_rate_by_age.append(RatioTimeSeries(name='Cumulative vaccination rate',
                                                   numerator_sum_time_series=cum_vaccine_by_age[0],
                                                   denominator_sum_time_series=pop_size_by_age[0],
                                                   if_surveyed=True))
    
    # incidence and new hospitalization by profile (dominant, novel, vaccinated)
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

    # incidence and new hospitalization among the vaccinated
    Is_vacc = []
    Hs_vacc = []
    Is_novel = []
    Hs_novel = []
    for p in range(age_groups_profiles.nProfiles):
        # find Is and Hs that are vaccinated
        if p in (Profiles.DOM_VAC.value, Profiles.NOV_VAC.value):
            for a in range(age_groups_profiles.nAgeGroups):
                i = age_groups_profiles.get_row_index(age_group=a, profile=p)
                Is_vacc.append(Is[i])
                Hs_vacc.append(Hs[i])
        # find Is and Hs that are due to novel variant
        if p in (Profiles.NOV_UNVAC.value, Profiles.NOV_VAC.value):
            for a in range(age_groups_profiles.nAgeGroups):
                i = age_groups_profiles.get_row_index(age_group=a, profile=p)
                Is_novel.append(Is[i])
                Hs_novel.append(Hs[i])

    incd_vacc = SumIncidence(
        name='Incidence-vaccinated', compartments=Is_vacc)
    new_hosp_vacc = SumIncidence(
        name='New hospitalizations and vaccinated', compartments=Hs_vacc)
    incd_novel = SumIncidence(
        name='Incidence-novel', compartments=Is_novel)
    new_hosp_novel = SumIncidence(
        name='New hospitalizations-novel', compartments=Hs_novel)
    perc_incd_vacc = RatioTimeSeries(name='% of incidence that are vaccinated',
                                     numerator_sum_time_series=incd_vacc,
                                     denominator_sum_time_series=incd_by_age[0],
                                     if_surveyed=True)
    perc_new_hosp_vacc = RatioTimeSeries(name='% of new hospitalizations that are vaccinated',
                                         numerator_sum_time_series=new_hosp_vacc,
                                         denominator_sum_time_series=new_hosp_by_age[0],
                                         if_surveyed=True)
    perc_incd_novel = RatioTimeSeries(name='% of incidence due to novel variant',
                                      numerator_sum_time_series=incd_novel,
                                      denominator_sum_time_series=incd_by_age[0],
                                      if_surveyed=True)
    perc_new_hosp_novel = RatioTimeSeries(name='% of new hospitalizations due to novel variant',
                                          numerator_sum_time_series=new_hosp_novel,
                                          denominator_sum_time_series=new_hosp_by_age[0],
                                          if_surveyed=True)

    # list to contain summation statistics for age groups
    for a in range(age_groups_profiles.nAgeGroups):
        str_a = age_groups_profiles.get_str_age(age_group=a)

        comparts_this_age = [Ss[a], Vs[a]]
        Is_this_age = []
        Hs_this_age = []
        Ds_this_age = []

        # number vaccinated
        vaccinated_this_age = [counting_vacc_in_S[a], counting_vacc_in_R_dom[a], counting_vacc_in_R_nov[a]]

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
            denominator_sum_time_series=pop_size_by_age[-1],
            if_surveyed=True))
        # cumulative incidence
        cum_incd_by_age.append(SumCumulativeIncidence(
            name='Cumulative incidence-' + str_a, compartments=Is_this_age))
        # age-distribution of cumulative incidence
        age_dist_cum_incd.append(RatioTimeSeries(
            name='Cumulative incidence-'+str_a+' (%)',
            numerator_sum_time_series=cum_incd_by_age[-1],
            denominator_sum_time_series=cum_incd_by_age[0],
            if_surveyed=True))

        # new hospitalizations
        new_hosp_by_age.append(SumIncidence(
            name='New hospitalizations-' + str_a, compartments=Hs_this_age))
        # rate of new hospitalizations
        new_hosp_rate_by_age.append(RatioTimeSeries(
            name='Hospitalization rate-'+str_a,
            numerator_sum_time_series=new_hosp_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1],
            if_surveyed=True))

        # cumulative hospitalizations
        cum_hosp_by_age.append(SumCumulativeIncidence(
            name='Cumulative hospitalizations-' + str_a, compartments=Hs_this_age))
        # rate of cumulative hospitalizations
        cum_hosp_rate_by_age.append(RatioTimeSeries(
            name='Cumulative hospitalization rate-' + str_a,
            numerator_sum_time_series=cum_hosp_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1],
            if_surveyed=True))
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
            name='Cumulative vaccination-' + str_a, compartments=vaccinated_this_age))
        # rate of cumulative vaccination
        cum_vaccine_rate_by_age.append(RatioTimeSeries(
            name='Cumulative vaccination rate-' + str_a,
            numerator_sum_time_series=cum_vaccine_by_age[-1],
            denominator_sum_time_series=pop_size_by_age[-1],
            if_surveyed=True))

    # --------- calibration and feasibility conditions ---------
    if sets.calcLikelihood:
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
                feasible_max=MAX_PREV_IMMUNE_FROM_INF/100,
                period=[0, FEASIBILITY_PERIOD]))

        prev_immune_from_inf.add_calibration_targets(
            ratios=sets.prevImmFromInfMean, variances=sets.prevImmFromInfVar)

        # calibration information for the overall hospitalization rate
        cum_hosp_rate_by_age[0].add_calibration_targets(
            ratios=sets.cumHospRateMean, variances=sets.cumHospRateVar)

        # calibration information for hospitalization rate by age
        for a in range(age_groups_profiles.nAgeGroups):
            cum_hosp_rate_by_age[a+1].add_calibration_targets(
                ratios=sets.cumHospRateByAgeMean[a], survey_sizes=sets.cumHospRateByAgeN[a])

        # calibration information for the overall vaccination coverage
        cum_vaccine_rate_by_age[0].add_calibration_targets(
            ratios=sets.cumVaccRateMean, survey_sizes=sets.cumVaccRateN)

        # calibration information for the percentage of infection associated with the novel variant
        perc_incd_novel.add_calibration_targets(
            ratios=sets.percInfWithNovelMean, survey_sizes=sets.percInfWithNovelN)

    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        settings=sets, params=params, in_hosp_rate=new_hosp_rate_by_age[0])

    # --------- populate the model ---------
    # change nodes
    chance_nodes = []
    chance_nodes.extend(ifs_hosp)
    chance_nodes.extend(ifs_novel_strain)
    chance_nodes.extend(counting_vacc_in_S)
    chance_nodes.extend(counting_vacc_in_R_dom)
    chance_nodes.extend(counting_vacc_in_R_nov)

    # summation-time series
    list_of_sum_time_series = []
    list_of_sum_time_series.extend(pop_size_by_age)
    list_of_sum_time_series.extend([num_susp, num_immune_from_inf])
    list_of_sum_time_series.extend(incd_by_age)
    list_of_sum_time_series.extend(new_hosp_by_age)
    list_of_sum_time_series.extend(cum_incd_by_age)
    list_of_sum_time_series.extend(cum_hosp_by_age)
    list_of_sum_time_series.extend(cum_death_by_age)
    list_of_sum_time_series.extend(cum_vaccine_by_age)
    list_of_sum_time_series.extend(incd_by_profile)
    list_of_sum_time_series.extend(new_hosp_by_profile)
    list_of_sum_time_series.extend([incd_vacc, new_hosp_vacc, incd_novel, new_hosp_novel])

    # ratio time-series
    list_of_ratio_time_series = []
    list_of_ratio_time_series.extend(incd_rate_by_age)
    list_of_ratio_time_series.extend([prev_susp, prev_immune_from_inf])
    list_of_ratio_time_series.extend(new_hosp_rate_by_age)
    list_of_ratio_time_series.extend(cum_hosp_rate_by_age)
    list_of_ratio_time_series.extend(cum_death_rate_by_age)
    list_of_ratio_time_series.extend(cum_vaccine_rate_by_age)
    list_of_ratio_time_series.extend(profile_dist_incd)
    list_of_ratio_time_series.extend(profile_dist_new_hosp)
    list_of_ratio_time_series.extend(age_dist_cum_incd)
    list_of_ratio_time_series.extend(age_dist_new_hosp)
    list_of_ratio_time_series.extend(age_dist_cum_death)
    list_of_ratio_time_series.extend([perc_incd_vacc, perc_new_hosp_vacc,
                                      perc_incd_novel, perc_new_hosp_novel])

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
    feature_on_pd = None

    # conditions
    pass_feas_period = None
    on_condition_during_y1 = None
    off_condition = None
    off_condition_during_y1 = None

    # interventions
    physical_dist = None

    # --------- features ---------
    # defined on surveyed in hospital
    feature_on_surveyed_hosp = FeatureSurveillance(name='Surveyed number in ICU',
                                                   ratio_time_series_with_surveillance=in_hosp_rate)
    # feature on time
    feature_on_epi_time = FeatureEpidemicTime(name='epidemic time')

    # if feasibility period has passed
    pass_feas_period = ConditionOnFeatures(
        name='if year {} has passed'.format(FEASIBILITY_PERIOD),
        features=[feature_on_epi_time],
        signs=['ge'],
        thresholds=[FEASIBILITY_PERIOD])

    # use of physical distancing during FEASIBILITY_PERIOD
    if settings.ifPDInCalibrationPeriod:
        # ---------- intervention -------
        physical_dist = InterventionAffectingContacts(
            name='Physical distancing during calibration period',
            par_perc_change_in_contact_matrix=params.matrixOfPercChangeInContactsY1)

        # --------- features ---------
        # feature defined on the intervention
        feature_on_pd = FeatureIntervention(name='Status of pd',
                                            intervention=physical_dist)
        # --------- conditions ---------
        on_condition_during_y1 = ConditionOnFeatures(
            name='turn on pd',
            features=[feature_on_epi_time, feature_on_pd, feature_on_surveyed_hosp],
            signs=['l', 'e', 'ge'],
            thresholds=[FEASIBILITY_PERIOD, 0, params.pdY1Thresholds[0]])
        off_condition = ConditionOnFeatures(
            name='turn off pd',
            features=[feature_on_pd, feature_on_surveyed_hosp],
            signs=['e', 'l'],
            thresholds=[1, params.pdY1Thresholds[1]])
        off_condition_during_y1 = ConditionOnConditions(
            name='turn off pd Y1',
            conditions=[pass_feas_period, off_condition],
            logic='or')

        # --------- decision rule ---------
        decision_rule = ConditionBasedDecisionRule(
            default_switch_value=0,
            condition_to_turn_on=on_condition_during_y1,
            condition_to_turn_off=off_condition_during_y1)
        physical_dist.add_decision_rule(decision_rule=decision_rule)

    # make the list of features, conditions, and interventions
    features = [feature_on_surveyed_hosp, feature_on_epi_time, feature_on_pd]
    conditions = [pass_feas_period, on_condition_during_y1, off_condition, off_condition_during_y1]

    interventions = [physical_dist]

    return interventions, features, conditions
