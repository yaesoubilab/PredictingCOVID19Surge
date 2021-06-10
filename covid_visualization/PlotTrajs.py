import apace.analysis.Trajectories as A
import covid_model.data as D
import definitions as Def
from definitions import AgeGroups, Profiles

A.FEASIBLE_REGION_COLOR_CODE = 'pink'


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

    # ------ plot information for the validation plot (by age) --------
    for a in range(indexer.nAgeGroups):

        str_a = indexer.get_str_age(age_group=a)
        S = A.TrajPlotInfo(outcome_name='In: Susceptible-'+str_a, title='Susceptible-'+str_a,
                           y_range=(0, 55000), x_multiplier=prev_multiplier)
        V = A.TrajPlotInfo(outcome_name='In: Vaccinated-'+str_a, title='Vaccinated-'+str_a,
                           y_range=(0, 55000), x_multiplier=prev_multiplier)

        Es = []
        Is = []
        Hs = []
        ICUs = []
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
            ICUs.append(A.TrajPlotInfo(outcome_name='In: ICU-'+str_a_p, title='ICU-'+str_a_p,
                                       y_range=(0, 250), x_multiplier=prev_multiplier))
            Rs.append(A.TrajPlotInfo(outcome_name='In: Recovered-'+str_a_p, title='Recovered-'+str_a_p,
                                     y_range=(0, 105000), x_multiplier=prev_multiplier))
            Ds.append(A.TrajPlotInfo(outcome_name='In: Death-'+str_a_p, title='Cumulative death-'+str_a_p,
                                     y_range=(0, 500), x_multiplier=prev_multiplier))

        # validation
        filename_validation = 'outputs/fig_trajs/{}.png'.format(str_a)
        sim_outcomes.plot_multi_panel(n_rows=3, n_cols=6,
                                      list_plot_info=[Es[0], Is[0], Hs[0], ICUs[0], Rs[0], Rs[0],
                                                      Es[1], Is[1], Hs[1], ICUs[1], Rs[1], Rs[1],
                                                      S, V],
                                      file_name=filename_validation,
                                      figure_size=(11, 5.5))

    # ------ plot information for the summary plot --------
    ObsIncA = A.TrajPlotInfo(outcome_name='Obs: Incidence-0', title='Incidence-A',
                             y_range=(0, 25000), x_multiplier=obs_incd_multiplier)
    ObsIncB = A.TrajPlotInfo(outcome_name='Obs: Incidence-1', title='Incidence-B',
                             y_range=(0, 25000), x_multiplier=obs_incd_multiplier)
    ObsInc = A.TrajPlotInfo(outcome_name='Obs: Incidence', title='Incidence',
                            y_range=(0, 25000), x_multiplier=obs_incd_multiplier)
    ObsICU = A.TrajPlotInfo(outcome_name='Obs: # in ICU', title='ICU Occupancy',
                            y_range=(0, 250), x_multiplier=prev_multiplier,
                            calibration_info=A.CalibrationTargetPlotInfo(
                                feasible_range_info=A.FeasibleRangeInfo(
                                    x_range=[0, SIM_DURATION], y_range=[0, 10.34])))
    ObsCaseFatality = A.TrajPlotInfo(outcome_name='Obs: Case fatality',
                                     title='Case-fatality (%)',
                                     y_range=(0, 100), y_multiplier=100,
                                     x_multiplier=obs_incd_multiplier)
    ObsPercB = A.TrajPlotInfo(outcome_name='Obs: % of cases infected with novel strain',
                              title='Cases with novel strain (%)',
                              y_range=(0, 100), y_multiplier=100,
                              x_multiplier=obs_incd_multiplier)

    ObsPercV = A.TrajPlotInfo(outcome_name='Obs: % of population vaccinated',
                              title='Vaccinated (%)',
                              y_range=(0,100), y_multiplier=100,
                              x_multiplier=prev_multiplier,
                              calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=D.VACCINE_COVERAGE))

    # summary
    filename_summary = 'outputs/fig_trajs/summary.png'
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=3,
                                  list_plot_info=[ObsInc, ObsIncA, ObsIncB, ObsICU, ObsPercB, ObsPercV],
                                  file_name=filename_summary,
                                  show_subplot_labels=True,
                                  figure_size=(6, 4)
                                  )

    # ------ plot information for the age-distribution of outcome --------
    age_dist_incd = []
    age_dist_in_hosp = []
    age_dist_in_icu = []
    age_dist_cum_death = []

    for a in range(indexer.nAgeGroups):

        str_a = indexer.get_str_age(age_group=a)

        age_dist_incd.append(A.TrajPlotInfo(outcome_name='Incidence-{} (%)'.format(str_a),
                                            title='Incidence-{} (%)'.format(str_a),
                                            y_range=(0, 100), y_multiplier=100, x_multiplier=incd_multiplier))
        age_dist_in_hosp.append(A.TrajPlotInfo(outcome_name='Hospitalized-{} (%)'.format(str_a),
                                               title='Hospitalized-{} (%)'.format(str_a),
                                               y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))
        age_dist_in_icu.append(A.TrajPlotInfo(outcome_name='In ICU-{} (%)'.format(str_a),
                                              title='In ICU-{} (%)'.format(str_a),
                                              y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))
        age_dist_cum_death.append(A.TrajPlotInfo(outcome_name='Cumulative death-{} (%)'.format(str_a),
                                                 title='Cumulative death-{} (%)'.format(str_a),
                                                 y_range=(0, 100), y_multiplier=100, x_multiplier=prev_multiplier))

    filename_validation = 'outputs/fig_trajs/age_dist.png'
    list_plot_info = age_dist_incd
    list_plot_info.extend(age_dist_in_hosp)
    list_plot_info.extend(age_dist_in_icu)
    list_plot_info.extend(age_dist_cum_death)
    sim_outcomes.plot_multi_panel(n_rows=4, n_cols=6,
                                  list_plot_info=list_plot_info,
                                  file_name=filename_validation,
                                  figure_size=(10, 7))
