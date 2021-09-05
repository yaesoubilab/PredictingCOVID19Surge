import apace.analysis.Trajectories as A
import covid_model.data as D
import definitions as Def
from covid_model.data import *
from covid_model.settings import COVIDSettings
from definitions import AgeGroups, Profiles, FEASIBILITY_PERIOD, ROOT_DIR


A.FEASIBLE_REGION_COLOR_CODE = 'pink'
IF_MAKE_VALIDATION_PLOTS = False
A.Y_LABEL_COORD_X = -0.15
A.SUBPLOT_W_SPACE = 0.0


def plot(prev_multiplier=52, incd_multiplier=1, obs_incd_multiplier=1, n_random_trajs_to_display=None):
    """
    :param prev_multiplier: (int) to multiply the simulation time to convert it to year, week, or day.
    :param incd_multiplier: (int) to multiply the simulation period to covert it to year, week, or day.
    :param obs_incd_multiplier: (int) to multiply the observation period to conver it to year, week, or day.
    :param n_random_trajs_to_display: (int) number of trajectories to display
    :return:
    """

    directory = ROOT_DIR + '/outputs/trajectories'
    sim_outcomes = A.SimOutcomeTrajectories(csv_directory=directory)

    # defaults
    SIM_DURATION = Def.SIM_DURATION*52
    A.X_RANGE = (0, SIM_DURATION)     # x-axis range
    A.X_TICKS = (0, 52/2)      # x-axis ticks (min at 0 with interval of 5)
    A.X_LABEL = 'Weeks since March 1, 2010'     # x-axis label

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
            filename_validation = ROOT_DIR+'/outputs/figures/{}.png'.format(str_a)
            sim_outcomes.plot_multi_panel(n_rows=3, n_cols=6,
                                          list_plot_info=[Es[0], Is[0], Hs[0], Rs[0], Rs[0],
                                                          Es[1], Is[1], Hs[1], Rs[1], Rs[1],
                                                          S, V],
                                          n_random_trajs_to_display=n_random_trajs_to_display,
                                          file_name=filename_validation,
                                          figure_size=(11, 5.5))

    # -----------------------------------------------------
    # ------ plot information for the summary plot --------
    # -----------------------------------------------------
    obs_inc_rate = A.TrajPlotInfo(
        outcome_name='Obs: Incidence rate',
        title='Incidence rate\n(per 100,000 population)',
        y_range=(0, 25000), y_multiplier=100000, x_multiplier=obs_incd_multiplier)
    obs_hosp_rate = A.TrajPlotInfo(
        outcome_name='Obs: New hospitalization rate',
        title='Hospitalization rate\n(per 100,000 population)',
        y_range=(0, MAX_HOSP_RATE_OVERALL * 4), y_multiplier=100000, x_multiplier=incd_multiplier,
        calibration_info=A.CalibrationTargetPlotInfo(
            feasible_range_info=A.FeasibleRangeInfo(
                x_range=[0, FEASIBILITY_PERIOD*52],
                y_range=[MIN_HOSP_RATE_OVERALL, MAX_HOSP_RATE_OVERALL])))
    obs_cum_hosp_rate = A.TrajPlotInfo(
        outcome_name='Obs: Cumulative hospitalization rate',
        title='Cumulative hospitalization rate\n(per 100,000 population)',
        y_range=(0, 1000*3), y_multiplier=100000, x_multiplier=prev_multiplier,
        calibration_info=A.CalibrationTargetPlotInfo(
            rows_of_data=CUM_HOSP_RATE_OVERALL
        ))
    obs_prev_immune_from_inf = A.TrajPlotInfo(
        outcome_name='Obs: Prevalence with immunity from infection',
        title='Prevalence of population with\nimmunity from infection (%)',
        y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
        calibration_info=A.CalibrationTargetPlotInfo(
            rows_of_data=PREV_IMMUNE_FROM_INF,
            feasible_range_info=A.FeasibleRangeInfo(
                x_range=[0, FEASIBILITY_PERIOD * 52],
                y_range=[0, MAX_PREV_IMMUNE_FROM_INF])))
    obs_cum_vacc_rate = A.TrajPlotInfo(
        outcome_name='Obs: Cumulative vaccination rate',
        title='Cumulative vaccination rate (%)',
        y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
        calibration_info=A.CalibrationTargetPlotInfo(
            rows_of_data=D.VACCINE_COVERAGE_OVER_TIME,
            if_connect_obss=True))
    obs_incd_novel = A.TrajPlotInfo(
        outcome_name='Obs: % of incidence with novel variant',
        title='Incidence associated with\n'
              'novel strain (%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)

    # summary
    sim_outcomes.plot_multi_panel(n_rows=1, n_cols=3,
                                  list_plot_info=[obs_hosp_rate, obs_cum_hosp_rate,
                                                  obs_cum_vacc_rate],
                                  file_name=ROOT_DIR+'/outputs/figures/summary3.png',
                                  n_random_trajs_to_display=n_random_trajs_to_display,
                                  show_subplot_labels=True,
                                  figure_size=(2.3*3, 2.4)
                                  )
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=3,
                                  list_plot_info=[obs_hosp_rate, obs_cum_hosp_rate, obs_prev_immune_from_inf,
                                                  obs_cum_vacc_rate, obs_incd_novel],
                                  file_name=ROOT_DIR+'/outputs/figures/summary.png',
                                  n_random_trajs_to_display=n_random_trajs_to_display,
                                  show_subplot_labels=True,
                                  figure_size=(2.3*3, 2.4*2)
                                  )

    # -----------------------------------------------------
    # ------ plot information for the novel variant plot --------
    # -----------------------------------------------------
    obs_incd_novel = A.TrajPlotInfo(
        outcome_name='Obs: % of incidence with novel variant',
        title='Incidence associated with\n'
              'novel strain (%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier,
        calibration_info=A.CalibrationTargetPlotInfo(
            rows_of_data=D.PERC_INF_WITH_NOVEL,
            if_connect_obss=False))
    obs_incd_novel_unvacc = A.TrajPlotInfo(
        outcome_name='Obs: % of incidence due to Novel-Unvaccinated',
        title='Incidence associated with\n'
              'novel variant among the unvaccinated\n(%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)
    obs_incd_novl_vacc = A.TrajPlotInfo(
        outcome_name='Obs: % of incidence due to Novel-Vaccinated',
        title='Incidence associated with\n'
              'novel variant among the vaccinated\n(%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)
    obs_new_hosp_novel = A.TrajPlotInfo(
        outcome_name='Obs: % of new hospitalizations with novel variant',
        title='New hospitalizations associated with\n'
              'novel variant (%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)
    obs_new_hosp_novel_unvacc = A.TrajPlotInfo(
        outcome_name='Obs: % of new hospitalizations due to Novel-Unvaccinated',
        title='New hospitalizations associated with\n'
              'novel strain among the unvaccinated\n(%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)
    obs_new_hosp_novel_vacc = A.TrajPlotInfo(
        outcome_name='Obs: % of new hospitalizations due to Novel-Vaccinated',
        title='New hospitalizations associated with\n'
              'novel strain among the vaccinated\n(%)',
        y_range=(0, 100), y_multiplier=100,
        x_multiplier=obs_incd_multiplier)

    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=3,
                                  list_plot_info=[obs_incd_novel, obs_incd_novel_unvacc, obs_incd_novl_vacc,
                                                  obs_new_hosp_novel, obs_new_hosp_novel_unvacc, obs_new_hosp_novel_vacc],
                                  file_name=ROOT_DIR+'/outputs/figures/novel_variant.png',
                                  n_random_trajs_to_display=n_random_trajs_to_display,
                                  show_subplot_labels=True,
                                  figure_size=(2.3*3, 2.4*2)
                                  )

    # -----------------------------------------------------
    # ------ plot information for calibration figure
    # -----------------------------------------------------
    hosp_rate_by_age = []
    cum_hosp_rate_by_age = []
    age_dist_cum_hosp = []
    cum_vaccine_rate_by_age = []

    for a in range(indexer.nAgeGroups):
        str_a = indexer.get_str_age(age_group=a)

        hosp_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Hospitalization rate-{}'.format(str_a),
            title=str_a, y_label='Hospitalization rate\n(per 100,000 population)' if a == 0 else None,
            y_range=(0, 1000), y_multiplier=100000, x_multiplier=incd_multiplier,
            # calibration_info=A.CalibrationTargetPlotInfo(
            #     feasible_range_info=A.FeasibleRangeInfo(
            #         x_range=[0, CALIB_PERIOD * 52], y_range=[0, MAX_HOSP_RATE_BY_AGE[a]]))
        ))

        cum_hosp_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Cumulative hospitalization rate-{}'.format(str_a),
            title=str_a, y_label='Cumulative hospitalization rate\n(per 100,000 population)' if a == 0 else None,
            y_range=(0, 5000), y_multiplier=100000, x_multiplier=prev_multiplier,
            calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=D.CUM_HOSP_RATE_BY_AGE[a])))

        age_dist_cum_hosp.append(A.TrajPlotInfo(
            outcome_name='Cumulative hospitalizations-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\ncumulative hospitalizations (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
            calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=D.HOSP_AGE_DIST[a])))

        cum_vaccine_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Cumulative vaccination rate-{}'.format(str_a),
            title=str_a, y_label='Cumulative vaccination rate (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier,
            calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=VACCINE_COVERAGE_BY_AGE[a],
                                                         if_connect_obss=True)
        ))

    filename_validation = ROOT_DIR+'/outputs/figures/calibration.png'
    list_plot_info = hosp_rate_by_age
    list_plot_info.extend(cum_hosp_rate_by_age)
    list_plot_info.extend(age_dist_cum_hosp)
    list_plot_info.extend(cum_vaccine_rate_by_age)
    A.Y_LABEL_COORD_X = -0.35
    sim_outcomes.plot_multi_panel(n_rows=4, n_cols=len(AgeGroups),
                                  list_plot_info=list_plot_info,
                                  n_random_trajs_to_display=n_random_trajs_to_display,
                                  file_name=filename_validation,
                                  figure_size=(11, 6.5))

    # --------------------------------------------------------------------
    # ------ plot information for the incidence figure  --------
    # --------------------------------------------------------------------
    incd_rate_by_age = []
    age_dist_cum_incd = []
    cum_death_rate_by_age = []
    age_dist_cum_death = []

    for a in range(indexer.nAgeGroups):

        str_a = indexer.get_str_age(age_group=a)

        incd_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Incidence rate-{}'.format(str_a),
            title=str_a, y_label='Incidence rate\n(per 100,000 population) ' if a == 0 else None,
            y_range=(0, 20000), y_multiplier=100000, x_multiplier=incd_multiplier))
        age_dist_cum_incd.append(A.TrajPlotInfo(
            outcome_name='Cumulative incidence-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\ncumulative incident (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))
        cum_death_rate_by_age.append(A.TrajPlotInfo(
            outcome_name='Cumulative death rate-{}'.format(str_a),
            title=str_a, y_label='Cumulative deaths rate\n(per 100,000 population)' if a == 0 else None,
            y_range=(0, 1000), y_multiplier=100000, x_multiplier=prev_multiplier))

        age_dist_cum_death.append(A.TrajPlotInfo(
            outcome_name='Cumulative death-{} (%)'.format(str_a),
            title=str_a, y_label='Age-distribution of\n cumulative deaths (%)' if a == 0 else None,
            y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))

    filename_validation = ROOT_DIR+'/outputs/figures/incidence.png'
    list_plot_info = incd_rate_by_age
    list_plot_info.extend(age_dist_cum_incd)
    #list_plot_info.extend(age_dist_cum_death)
    A.Y_LABEL_COORD_X = -0.25
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=len(AgeGroups),
                                  list_plot_info=list_plot_info,
                                  n_random_trajs_to_display=n_random_trajs_to_display,
                                  file_name=filename_validation,
                                  figure_size=(15, 4))


if __name__ == "__main__":

    # get model settings
    sets = COVIDSettings()

    plot(prev_multiplier=52,  # to show weeks on the x-axis of prevalence data
         incd_multiplier=sets.simulationOutputPeriod * 52,  # to show weeks on the x-axis of incidence data
         obs_incd_multiplier=sets.observationPeriod * 52,
         )
