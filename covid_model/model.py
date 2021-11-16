from SimPy.Parameters import Constant
from apace.ModelObjects import Compartment, ChanceNode, Counter, \
    DeathCompartment, EpiIndepEvent, EpiDepEvent, PoissonEvent
from apace.TimeSeries import SumIncidence, SumPrevalence, SumCumulativeIncidence, RatioTimeSeries
from covid_model.model_support import get_interventions_features_conditions, add_calibration_info
from covid_model.parameters import COVIDParameters
from definitions import ProfileDefiner, Variants


def build_covid_model(model):
    """ populates the provided model with a COVID model.
    :param model: an empty EpiModel to be populated
    """

    # store the model settings in a local variable for faster processing
    sets = model.settings

    # parameters of the COVID model
    params = COVIDParameters(novel_variant_will_emerge=sets.novelVariantWillEmerge)

    pd = ProfileDefiner(n_age_groups=params.nAgeGroups, n_variants=params.nVariants, n_vacc_status=2)

    Ss = [None] * pd.nAgeGroups
    Vs = [None] * pd.nAgeGroups
    Es = [None] * pd.length
    Is = [None] * pd.length
    Hs = [None] * pd.length
    Rs = [None] * pd.length
    Ds = [None] * pd.length
    ifs_hosp = [None] * pd.length
    counting_vacc_in_S = [None] * pd.nAgeGroups
    counting_vacc_in_R_by_variant = [[None]*pd.nVariants for a in range(pd.nAgeGroups)]

    # events
    importation_by_variant = [[None]*pd.nVariants for a in range(pd.nAgeGroups)]
    infection_in_S_or_V = [None] * pd.length
    infection_in_R_by_variant = [[None] * pd.nVariants for i in range(pd.length)]
    leaving_Es = [None] * pd.length
    leaving_Is = [None] * pd.length
    leaving_Hs = [None] * pd.length
    leaving_Rs = [None] * pd.length
    deaths_in_hosp = [None] * pd.length
    vaccination_in_S = [None] * pd.nAgeGroups
    vaccination_in_R_by_variant = [[None]*pd.nVariants for a in range(pd.nAgeGroups)]
    losing_vaccine_immunity = [None] * pd.nAgeGroups

    # --------- model compartments ---------
    for a in range(pd.nAgeGroups):
        str_a = pd.strAge[a]
        Ss[a] = Compartment(name='Susceptible-'+str_a, size_par=params.sizeSByAge[a],
                            susceptibility_params=[Constant(1), Constant(1), Constant(1)],
                            row_index_contact_matrix=a)
        Vs[a] = Compartment(name='Vaccinated-'+str_a,
                            susceptibility_params=params.suspVaccByVariant,
                            row_index_contact_matrix=a)

        for v in range(pd.nVariants):
            for vs in range(pd.nVaccStatus):
                str_a_p = pd.strAgeProfile[a][v][vs]
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                p = pd.get_profile_index(variant=v, vacc_status=vs)

                # infectivity and susceptibility
                inf_params = [Constant(0), Constant(0), Constant(0)]
                inf_params[v] = params.infectivityByVaccByVariant[vs][v]

                # -------- compartments ----------
                Es[i] = Compartment(name='Exposed-'+str_a_p,
                                    num_of_pathogens=pd.nVariants, if_empty_to_eradicate=True,
                                    row_index_contact_matrix=a)
                Is[i] = Compartment(name='Infectious-'+str_a_p,
                                    size_par=params.sizeIProfile0ByAge[a] if v == 0 else Constant(value=0),
                                    infectivity_params=inf_params, if_empty_to_eradicate=True,
                                    row_index_contact_matrix=a)
                Hs[i] = Compartment(name='Hospitalized-'+str_a_p,
                                    num_of_pathogens=pd.nVariants, if_empty_to_eradicate=True,
                                    row_index_contact_matrix=a)
                Rs[i] = Compartment(name='Recovered-'+str_a_p,
                                    susceptibility_params=params.suspInRByProfileByVariant[p],
                                    row_index_contact_matrix=a)
                Ds[i] = DeathCompartment(name='Death-'+str_a_p)

                # --------- chance nodes ---------
                # chance node to decide if an infected individual would get hospitalized
                ifs_hosp[i] = ChanceNode(name='If hospitalized-'+str_a_p,
                                         destination_compartments=[Hs[i], Rs[i]],
                                         probability_params=params.probHospByAgeAndProfile[a][p])

        # count vaccinations among recovered after infection with dominant or novel variants
        for v in range(pd.nVariants):
            str_a_p = pd.strAgeProfile[a][v][0]
            dest_after_vacc_in_recovered = pd.get_row_index(
                age_group=a, variant=v, vacc_status=1)
            counting_vacc_in_R_by_variant[a][v] = Counter(
                name='Vaccination in R-' + str_a_p,
                destination_compartment=Rs[dest_after_vacc_in_recovered])

        # count vaccinations among susceptibles
        counting_vacc_in_S[a] = Counter(name='Vaccination in S-' + str_a,
                                        destination_compartment=Vs[a])

        # --------- model outputs to collect ---------
        # set up prevalence, incidence, and cumulative incidence to collect
        if sets.ifCollectTrajsOfCompartments:

            Ss[a].setup_history(collect_prev=True)
            Vs[a].setup_history(collect_prev=True)

            for v in range(pd.nVariants):
                for vs in range(pd.nVaccStatus):
                    i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                    Es[i].setup_history(collect_prev=True)
                    Is[i].setup_history(collect_prev=True)
                    Hs[i].setup_history(collect_prev=True)
                    Rs[i].setup_history(collect_prev=True)
                    Ds[i].setup_history(collect_cum_incd=True)

        # --------- model events ---------
        for v in range(pd.nVariants):

            # infection in S and V
            s_idx = pd.get_row_index(age_group=a, variant=v, vacc_status=0)
            v_idx = pd.get_row_index(age_group=a, variant=v, vacc_status=1)
            infection_in_S_or_V[s_idx] = EpiDepEvent(
                name='Infection in S-' + str_a, destination=Es[s_idx], generating_pathogen=v)
            infection_in_S_or_V[v_idx] = EpiDepEvent(
                name='Infection in V-' + str_a, destination=Es[v_idx], generating_pathogen=v)

            # other events
            for vs in range(pd.nVaccStatus):
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                p = pd.get_profile_index(variant=v, vacc_status=vs)
                str_a_p = pd.strAgeProfile[a][v][vs]

                # infection in R
                for v_prime in range(pd.nVariants):
                    # if being infected with a novel variant
                    if v != v_prime and vs == 0:
                        i_prime = pd.get_row_index(age_group=a, variant=v_prime, vacc_status=vs)
                        infection_in_R_by_variant[i][v_prime] = EpiDepEvent(
                            name='Infection in R-' + str_a_p, destination=Es[i_prime], generating_pathogen=v_prime)

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

        # importation of new cases
        for v in range(pd.nVariants):
            str_a_v = pd.strAgeVariant[a][v]
            dest_if_novel = pd.get_row_index(age_group=a, variant=v, vacc_status=0)

            importation_by_variant[a][v] = PoissonEvent(
                name='Importation-'+str_a_v,
                destination=Es[dest_if_novel],
                rate_param=params.importRateByVariant[v])

        # vaccination in R
        for v in range(pd.nVariants):
            str_a_v = pd.strAgeVariant[a][v]
            vaccination_in_R_by_variant[a][v] = EpiIndepEvent(
                name='Vaccinating R-' + str_a_v,
                rate_param=params.vaccRateByAge[a],
                destination=counting_vacc_in_R_by_variant[a][v])

        vaccination_in_S[a] = EpiIndepEvent(
            name='Vaccinating S-'+str_a, rate_param=params.vaccRateByAge[a], destination=counting_vacc_in_S[a])

        losing_vaccine_immunity[a] = EpiIndepEvent(
            name='Losing vaccine immunity-'+str_a, rate_param=params.rateOfLosingVacImmunity, destination=Ss[a])

        # --------- connections of events and compartments ---------
        # attached epidemic events to compartments
        i_inf_org = pd.get_row_index(age_group=a, variant=Variants.ORIGINAL.value, vacc_status=0)
        i_inf_delta = pd.get_row_index(age_group=a, variant=Variants.DELTA.value, vacc_status=0)
        i_inf_novel = pd.get_row_index(age_group=a, variant=Variants.NOVEL.value, vacc_status=0)
        Ss[a].add_events(events=[infection_in_S_or_V[i_inf_org],  # infection with dominant
                                 infection_in_S_or_V[i_inf_delta],  # infection with delta
                                 infection_in_S_or_V[i_inf_novel],  # infection with novel
                                 vaccination_in_S[a],
                                 importation_by_variant[a][Variants.ORIGINAL.value],
                                 importation_by_variant[a][Variants.DELTA.value],
                                 importation_by_variant[a][Variants.NOVEL.value]])

        i_inf_org = pd.get_row_index(age_group=a, variant=Variants.ORIGINAL.value, vacc_status=1)
        i_inf_delta = pd.get_row_index(age_group=a, variant=Variants.DELTA.value, vacc_status=1)
        i_inf_novel = pd.get_row_index(age_group=a, variant=Variants.NOVEL.value, vacc_status=1)
        Vs[a].add_events(events=[infection_in_S_or_V[i_inf_org],  # infection with dominant
                                 infection_in_S_or_V[i_inf_delta],  # infection with delta
                                 infection_in_S_or_V[i_inf_novel],  # infection with novel
                                 losing_vaccine_immunity[a],
                                 ])

        for v in range(pd.nVariants):
            for vs in range(pd.nVaccStatus):
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                Es[i].add_event(event=leaving_Es[i])
                Is[i].add_event(event=leaving_Is[i])
                Hs[i].add_events(events=[leaving_Hs[i], deaths_in_hosp[i]])
                Rs[i].add_events(events=[leaving_Rs[i]])

                # add vaccination if necessary
                if vs == 0:
                    Rs[i].add_events(events=[vaccination_in_R_by_variant[a][v]])

                # add infection
                for v_prime in range(pd.nVariants):
                    if infection_in_R_by_variant[i][v_prime] is not None:
                        Rs[i].add_event(event=infection_in_R_by_variant[i][v_prime])

    # --------- sum time-series ------
    # population size
    compartments = Ss + Vs + Es + Is + Hs + Rs + Ds

    # lists to contain summation statistics
    # counts
    all_vaccinations = counting_vacc_in_S
    for c in counting_vacc_in_R_by_variant:
        all_vaccinations += c

    # unvaccinated individuals with immunity from infection
    unvacc_immune_from_infection = []
    for a in range(pd.nAgeGroups):
        for v in range(pd.nVariants):
            i = pd.get_row_index(age_group=a, variant=v, vacc_status=0)
            unvacc_immune_from_infection.append(Rs[i])

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
    hosp_occupancy = SumPrevalence(
        name='Hospital occupancy', compartments=Hs)
    num_susp = SumPrevalence(
        name='Individuals susceptible', compartments=Ss)
    num_immune_from_inf = SumPrevalence(
        name='Unvaccinated individuals wth immunity from infection',
        compartments=unvacc_immune_from_infection)
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

    hosp_occupancy_rate = RatioTimeSeries(name='Hospital occupancy rate',
                                          numerator_sum_time_series=hosp_occupancy,
                                          denominator_sum_time_series=pop_size_by_age[0],
                                          if_surveyed=True)
    params.y1EffOfControlMeasures.assign_sim_output(sim_output=hosp_occupancy_rate)
    params.y2EffOfControlMeasures.assign_sim_output(sim_output=hosp_occupancy_rate)

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
    for v in range(pd.nVariants):
        for vs in range(pd.nVaccStatus):

            # find Is and Hs in this profile
            Is_this_profile = []
            Hs_this_profile = []
            for a in range(pd.nAgeGroups):
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                Is_this_profile.append(Is[i])
                Hs_this_profile.append(Hs[i])

            str_profile = pd.strProfile[v][vs]

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
    for a in range(pd.nAgeGroups):
        for v in range(pd.nVariants):
            i = pd.get_row_index(age_group=a, variant=v, vacc_status=1)
            Is_vacc.append(Is[i])
            Hs_vacc.append(Hs[i])
    incd_vacc = SumIncidence(
        name='Incidence-vaccinated', compartments=Is_vacc)
    new_hosp_vacc = SumIncidence(
        name='New hospitalizations and vaccinated', compartments=Hs_vacc)
    perc_incd_vacc = RatioTimeSeries(name='% of incidence that are vaccinated',
                                     numerator_sum_time_series=incd_vacc,
                                     denominator_sum_time_series=incd_by_age[0],
                                     if_surveyed=True)
    perc_new_hosp_vacc = RatioTimeSeries(name='% of new hospitalizations that are vaccinated',
                                         numerator_sum_time_series=new_hosp_vacc,
                                         denominator_sum_time_series=new_hosp_by_age[0],
                                         if_surveyed=True)

    # incidence and new hospitalization due to delta and novel variant
    Is_by_variant = [[] for v in range(pd.nVariants)]
    Hs_by_variant = [[] for v in range(pd.nVariants)]
    for v in range(pd.nVariants):
        for a in range(pd.nAgeGroups):
            for vs in range(pd.nVaccStatus):
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
                Is_by_variant[v].append(Is[i])
                Hs_by_variant[v].append(Hs[i])

    incd_by_variant = [None] * pd.nVariants
    new_hosp_novel_by_variant = [None] * pd.nVariants
    for v in range(pd.nVariants):
        incd_by_variant[v] = SumIncidence(
            name='Incidence-'+pd.strVariant[v], compartments=Is_by_variant[v])
        new_hosp_novel_by_variant[v] = SumIncidence(
            name='New hospitalizations-'+pd.strVariant[v], compartments=Hs_by_variant[v])

    # percentage of incidence and new hospitalizations due to delta and novel variant
    perc_incd_by_variant = [None] * pd.nVariants
    perc_new_hosp_by_variant = [None] * pd.nVariants
    for v in range(pd.nVariants):
        perc_incd_by_variant[v] = RatioTimeSeries(
            name='% of incidence due to '+pd.strVariant[v],
            numerator_sum_time_series=incd_by_variant[v],
            denominator_sum_time_series=incd_by_age[0],
            if_surveyed=True)
        perc_new_hosp_by_variant[v] = RatioTimeSeries(
            name='% of new hospitalizations due to '+pd.strVariant[v],
            numerator_sum_time_series=new_hosp_novel_by_variant[v],
            denominator_sum_time_series=new_hosp_by_age[0],
            if_surveyed=True)

    # list to contain summation statistics for age groups
    for a in range(pd.nAgeGroups):
        str_a = pd.strAge[a]

        comparts_this_age = [Ss[a], Vs[a]]
        Is_this_age = []
        Hs_this_age = []
        Ds_this_age = []

        # number vaccinated
        vaccinated_this_age = [counting_vacc_in_S[a]] + counting_vacc_in_R_by_variant[a]

        for v in range(pd.nVariants):
            for vs in range(pd.nVaccStatus):
                i = pd.get_row_index(age_group=a, variant=v, vacc_status=vs)
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
            name='New hospitalization rate-'+str_a,
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
        add_calibration_info(settings=sets,
                             age_groups_profiles=pd,
                             hosp_occupancy_rate=hosp_occupancy_rate,
                             new_hosp_rate_by_age=new_hosp_rate_by_age,
                             prev_immune_from_inf=prev_immune_from_inf,
                             cum_hosp_rate_by_age=cum_hosp_rate_by_age,
                             cum_vaccine_rate_by_age=cum_vaccine_rate_by_age,
                             perc_incd_delta=perc_incd_by_variant[Variants.DELTA.value],
                             perc_incd_novel=perc_incd_by_variant[Variants.NOVEL.value])

    # --------- interventions, features, conditions ---------
    interventions, features, conditions = get_interventions_features_conditions(
        params=params,
        hosp_occupancy_rate=hosp_occupancy_rate,
        mitigating_strategies_on=sets.mitigatingStrategiesOn)

    # --------- populate the model ---------
    # change nodes
    chance_nodes = []
    chance_nodes.extend(ifs_hosp)
    chance_nodes.extend(counting_vacc_in_S)
    for c in counting_vacc_in_R_by_variant:
        chance_nodes.extend(c)

    # summation-time series
    list_of_sum_time_series = []
    list_of_sum_time_series.extend(pop_size_by_age)
    list_of_sum_time_series.append(hosp_occupancy)
    list_of_sum_time_series.extend([num_susp, num_immune_from_inf])
    list_of_sum_time_series.extend(incd_by_age)
    list_of_sum_time_series.extend(new_hosp_by_age)
    list_of_sum_time_series.extend(cum_incd_by_age)
    list_of_sum_time_series.extend(cum_hosp_by_age)
    list_of_sum_time_series.extend(cum_death_by_age)
    list_of_sum_time_series.extend(cum_vaccine_by_age)
    list_of_sum_time_series.extend(incd_by_profile)
    list_of_sum_time_series.extend(new_hosp_by_profile)
    list_of_sum_time_series.extend([incd_vacc, new_hosp_vacc])
    list_of_sum_time_series.extend(incd_by_variant)
    list_of_sum_time_series.extend(new_hosp_novel_by_variant)

    # ratio time-series
    list_of_ratio_time_series = []
    list_of_ratio_time_series.append(hosp_occupancy_rate)
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
    list_of_ratio_time_series.extend([perc_incd_vacc, perc_new_hosp_vacc])
    list_of_ratio_time_series.extend(perc_incd_by_variant)
    list_of_ratio_time_series.extend(perc_new_hosp_by_variant)

    model.populate(compartments=compartments,
                   parameters=params,
                   param_base_contact_matrix=params.baseContactMatrix,
                   chance_nodes=chance_nodes,
                   list_of_sum_time_series=list_of_sum_time_series,
                   list_of_ratio_time_series=list_of_ratio_time_series,
                   interventions=interventions,
                   features=features,
                   conditions=conditions)
