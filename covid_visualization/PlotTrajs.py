import apace.analysis.Trajectories as A
import covid_model.data as D
import definitions as Def
from covid_model.data import MAX_HOSP_RATE, MAX_HOSP_RATE_BY_AGE, VACCINE_COVERAGE_BY_AGE, HOSP_AGE_DIST
from definitions import AgeGroups, Profiles, CALIB_PERIOD

A.FEASIBLE_REGION_COLOR_CODE = 'pink'
IF_MAKE_VALIDATION_PLOTS = False


def plot(prev_multiplier=52, incd_multiplier=1, obs_incd_multiplier=1):
    """
    :param prev_multiplier: (int) to multiply the simulation time to convert it to year, week, or day.
    :param incd_multiplier: (int) to multiply the simulation period to covert it to year, week, or day.
    :return:
    """

    directory = 'outputs/trajectories'
    sim_outcomes = A.SimOutcomeTrajectories(csv_directory=directory)

    # defaults
    SIM_DURATION = Def.SIM_DURATION*52
    A.X_RANGE = (0, SIM_DURATION)     # x-axis range
    A.X_TICKS = (0, 52/2)      # x-axis ticks (min at 0 with interval of 5)
    A.X_LABEL = 'Weeks'     # x-axis label

    indexer = Def.AgeGroupsProfiles(n_age_groups=len(AgeGroups), n_profiles=len(Profiles))

    # -----------------------------------------------------------------
    # ------ plot information for the validation plot (by age) --------
    # -----------------------------------------------------------------
    if IF_MAKE_VALIDATION_PLOTS:
        for a in range(indexer.nAgeGroups):

            str_a = indexer.get_str_age(age_group=a)
            S = A.TrajPlotInfo(outcome_name='In: Susceptible-'+str_a, title='Susceptible-'+str_a,
                               y_range=(0, 55000), x_multiplier=prev_multiplier)
            V = A.TrajPlotInfo(outcome_name='In: Vaccinated-'+str_a, title='Vaccinated-'+str_a,
                               y_range=(0, 55000), x_multiplier=prev_multiplier)

            Es = []
            Is = []
            Hs = []
            Rs = []
            Ds = []
            for p in range(indexer.nProfiles):
                str_a_p = indexer.get_str_age_profile(age_group=a, profile=p)
                Es.append(A.TrajPlotInfo(outcome_name='In: Exposed-'+str_a_p, title='Exposed-'+str_a_p,
                                         y_range=(0, 22000), x_multiplier=prev_multiplier))
                Is.append(A.TrajPlotInfo(outcome_name='In: Infectious-'+str_a_p, title='Infectious-'+str_a_p,
                                         y_range=(0, 17000), x_multiplier=prev_multiplier))
                Hs.append(A.TrajPlotInfo(outcome_name='In: Hospitalized-'+str_a_p, title='Hospitalized-'+str_a_p,
                                         y_range=(0, 5000), x_multiplier=prev_multiplier))
                Rs.append(A.TrajPlotInfo(outcome_name='In: Recovered-'+str_a_p, title='Recovered-'+str_a_p,
                                         y_range=(0, 105000), x_multiplier=prev_multiplier))
                Ds.append(A.TrajPlotInfo(outcome_name='In: Death-'+str_a_p, title='Cumulative death-'+str_a_p,
                                         y_range=(0, 500), x_multiplier=prev_multiplier))

            # validation
            filename_validation = 'outputs/fig_trajs/{}.png'.format(str_a)
            sim_outcomes.plot_multi_panel(n_rows=3, n_cols=6,
                                          list_plot_info=[Es[0], Is[0], Hs[0], Rs[0], Rs[0],
                                                          Es[1], Is[1], Hs[1], Rs[1], Rs[1],
                                                          S, V],
                                          file_name=filename_validation,
                                          figure_size=(11, 5.5))

    # -----------------------------------------------------
    # ------ plot information for the summary plot --------
    # -----------------------------------------------------
    obs_inc_rate = A.TrajPlotInfo(outcome_name='Obs: Incidence rate',
                                  title='Incidence rate\n(per 100,000 population)',
                                  y_range=(0, 25000), y_multiplier=100000, x_multiplier=obs_incd_multiplier)
    obs_hosp_rate = A.TrajPlotInfo(outcome_name='Obs: New hospitalization rate',
                                   title='Hospitalization rate\n(per 100,000 population)',
                                   y_range=(0, MAX_HOSP_RATE * 4), y_multiplier=100000, x_multiplier=incd_multiplier,
                                   calibration_info=A.CalibrationTargetPlotInfo(
                                       feasible_range_info=A.FeasibleRangeInfo(
                                           x_range=[0, CALIB_PERIOD*52], y_range=[0, MAX_HOSP_RATE])))
    obs_new_hosp_novel = A.TrajPlotInfo(outcome_name='Obs: % of new hospitalizations due to Novel',
                                        title='New hospitalizations\nwith novel strain (%)',
                                        y_range=(0, 100), y_multiplier=100,
                                        x_multiplier=obs_incd_multiplier)
    obs_cum_vacc_rate = A.TrajPlotInfo(outcome_name='Obs: Cumulative vaccination rate',
                                       title='Cumulative vaccination rate (%)',
                                       y_range=(0, 100), y_multiplier=100,
                                       x_multiplier=prev_multiplier,
                                       calibration_info=A.CalibrationTargetPlotInfo(
                                           rows_of_data=D.VACCINE_COVERAGE_OVER_TIME))

    # summary
    filename_summary = 'outputs/fig_trajs/summary.png'
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=2,
                                  list_plot_info=[obs_inc_rate, obs_hosp_rate, obs_new_hosp_novel, obs_cum_vacc_rate],
                                  file_name=filename_summary,
                                  show_subplot_labels=True,
                                  figure_size=(5, 5)
                                  )

    # -----------------------------------------------------
    # ------ plot information for the rates by age --------
    # -----------------------------------------------------
    incd_rate_by_age = []
    hosp_rate_by_age = []
    cum_death_rate_by_age = []
    cum_vaccine_rate_by_age = []

    for a in range(indexer.nAgeGroups):
        str_a = indexer.get_str_age(age_group=a)

        incd_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Incidence rate-{}'.format(str_a),
            title=str_a, y_label='Incidence rate\n(per 100,000 population) ' if a == 0 else None,
            y_range=(0, 20000), y_multiplier=100000, x_multiplier=incd_multiplier))
        hosp_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Hospitalization rate-{}'.format(str_a),
            title=str_a, y_label='Hospitalization rate\n(per 100,000 population)' if a == 0 else None,
            y_range=(0, 1000), y_multiplier=100000, x_multiplier=incd_multiplier,
            calibration_info=A.CalibrationTargetPlotInfo(
                feasible_range_info=A.FeasibleRangeInfo(
                    x_range=[0, CALIB_PERIOD * 52], y_range=[0, MAX_HOSP_RATE_BY_AGE[a]]))))
        cum_death_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Cumulative death rate-{}'.format(str_a),
            title=str_a, y_label='Cumulative deaths rate\n(per 100,000 population)' if a == 0 else None,
            y_range=(0, 1000), y_multiplier=100000, x_multiplier=prev_multiplier))
        cum_vaccine_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Cumulative vaccination rate-{}'.format(str_a),
            title=str_a, y_label='Cumulative vaccination rate (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
            calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=VACCINE_COVERAGE_BY_AGE[a])
        ))

    filename_validation = 'outputs/fig_trajs/rates_by_age.png'
    list_plot_info = incd_rate_by_age
    list_plot_info.extend(hosp_rate_by_age)
    #list_plot_info.extend(cum_death_rate_by_age)
    list_plot_info.extend(cum_vaccine_rate_by_age)
    A.Y_LABEL_COORD_X = -0.35
    sim_outcomes.plot_multi_panel(n_rows=3, n_cols=6,
                                  list_plot_info=list_plot_info,
                                  file_name=filename_validation,
                                  figure_size=(9.5, 5))

    # --------------------------------------------------------------------
    # ------ plot information for the age-distribution of outcome --------
    # --------------------------------------------------------------------
    age_dist_cum_incd = []
    age_dist_cum_hosp = []
    age_dist_cum_death = []

    for a in range(indexer.nAgeGroups):

        str_a = indexer.get_str_age(age_group=a)

        # calibration info for age distribution
        if a in (AgeGroups.Age_65_74.value, AgeGroups.Age_75_.value):
            calib_info = None
        else:
            calib_info = A.CalibrationTargetPlotInfo(rows_of_data=HOSP_AGE_DIST[a])

        age_dist_cum_incd.append(A.TrajPlotInfo(
            outcome_name='Cumulative incidence-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\ncumulative incident (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))
        age_dist_cum_hosp.append(A.TrajPlotInfo(
            outcome_name='Cumulative hospitalizations-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\ncumulative hospitalizations (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
            calibration_info=calib_info))
        age_dist_cum_death.append(A.TrajPlotInfo(
            outcome_name='Cumulative death-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\n cumulative deaths (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))


    filename_validation = 'outputs/fig_trajs/age_dist.png'
    list_plot_info = age_dist_cum_incd
    list_plot_info.extend(age_dist_cum_hosp)
    #list_plot_info.extend(age_dist_cum_death)
    A.Y_LABEL_COORD_X = -0.25
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=6,
                                  list_plot_info=list_plot_info,
                                  file_name=filename_validation,
                                  figure_size=(9.5, 3.6))
