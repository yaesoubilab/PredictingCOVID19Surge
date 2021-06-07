import covid_model.COVIDData as D
import apace.analysis.Trajectories as A

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
    SIM_DURATION = 2.25*52
    A.X_RANGE = (0, SIM_DURATION)     # x-axis range
    A.X_TICKS = (0, 52/2)      # x-axis ticks (min at 0 with interval of 5)
    A.X_LABEL = 'Weeks'     # x-axis label

    # plot information
    S = A.TrajPlotInfo(outcome_name='In: Susceptible', title='Susceptible',
                       y_range=(0, 105000), x_multiplier=prev_multiplier)
    V_Sus = A.TrajPlotInfo(outcome_name='In: Vaccinated-Susceptible', title='Vaccinated-Susceptible',
                           y_range=(0, 105000), x_multiplier=prev_multiplier)
    V_Imn = A.TrajPlotInfo(outcome_name='In: Vaccinated-Immune', title='Vaccinated-Immune',
                           y_range=(0, 105000), x_multiplier=prev_multiplier)

    E_A = A.TrajPlotInfo(outcome_name='In: Exposed-0', title='Exposed-A',
                         y_range=(0, 22000), x_multiplier=prev_multiplier)
    E_B = A.TrajPlotInfo(outcome_name='In: Exposed-1', title='Exposed-B',
                         y_range=(0, 22000), x_multiplier=prev_multiplier)
    I_A = A.TrajPlotInfo(outcome_name='In: Infectious-0', title='Infectious-A',
                         y_range=(0, 17000), x_multiplier=prev_multiplier)
    I_B = A.TrajPlotInfo(outcome_name='In: Infectious-1', title='Infectious-B',
                         y_range=(0, 17000), x_multiplier=prev_multiplier)
    H_A = A.TrajPlotInfo(outcome_name='In: Hospitalized-0', title='Hospitalized-A',
                         y_range=(0, 5000), x_multiplier=prev_multiplier)
    H_B = A.TrajPlotInfo(outcome_name='In: Hospitalized-1', title='Hospitalized-B',
                         y_range=(0, 5000), x_multiplier=prev_multiplier)
    ICU_A = A.TrajPlotInfo(outcome_name='In: ICU-0', title='In ICU-A',
                           y_range=(0, 250), x_multiplier=prev_multiplier)
    ICU_B = A.TrajPlotInfo(outcome_name='In: ICU-1', title='In ICU-B',
                           y_range=(0, 250), x_multiplier=prev_multiplier)
    R_A = A.TrajPlotInfo(outcome_name='In: Recovered-0', title='Recovered-A',
                         y_range=(0, 105000), x_multiplier=prev_multiplier)
    R_B = A.TrajPlotInfo(outcome_name='In: Recovered-1', title='Recovered-B',
                         y_range=(0, 105000), x_multiplier=prev_multiplier)
    D_A = A.TrajPlotInfo(outcome_name='Total to: Death-0', title='Cumulative deaths-A',
                         y_range=(0, 500), x_multiplier=prev_multiplier)
    D_B = A.TrajPlotInfo(outcome_name='Total to: Death-1', title='Cumulative deaths-B',
                         y_range=(0, 500), x_multiplier=prev_multiplier)

    Inc_A = A.TrajPlotInfo(outcome_name='To: Infectious-0', title='Incidence-A',
                           y_range=(0, 25000), x_multiplier=incd_multiplier)
    Inc_B = A.TrajPlotInfo(outcome_name='To: Infectious-1', title='Incidence-B',
                           y_range=(0, 25000), x_multiplier=incd_multiplier)

    Inc = A.TrajPlotInfo(outcome_name='Incidence', title='Incidence',
                         y_range=(0, 25000), x_multiplier=incd_multiplier)
    ICU = A.TrajPlotInfo(outcome_name='# in ICU', title='ICU Occupancy',
                         y_range=(0, 200), x_multiplier=prev_multiplier)
    PercB = A.TrajPlotInfo(outcome_name='% of cases infected with strain B', title='% cases with novel strain',
                           y_range=(0, 100), x_multiplier=incd_multiplier, y_multiplier=100)

    # surveyed measures
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
    ObsPercB = A.TrajPlotInfo(outcome_name='Obs: % of cases infected with strain B',
                              title='Cases with novel strain (%)',
                              y_range=(0, 100), y_multiplier=100,
                              x_multiplier=obs_incd_multiplier)

    ObsPercV = A.TrajPlotInfo(outcome_name='Obs: % of population vaccinated',
                              title='Vaccinated (%)',
                              y_range=(0,100), y_multiplier=100,
                              x_multiplier=prev_multiplier,
                              calibration_info=A.CalibrationTargetPlotInfo(rows_of_data=D.VACCINE_COVERAGE))

    filename_validation = 'figures/trajs (val).png'
    filename_summary = 'figures/trajs(sum).png'

    # validation
    sim_outcomes.plot_multi_panel(n_rows=4, n_cols=4,
                                  list_plot_info=[E_A, I_A, H_A, ICU_A,# R_A, D_A,
                                                  E_B, I_B, H_B, ICU_B,# R_B, D_B,
                                                  S, V_Imn, V_Sus, ICU, PercB,
                                                  Inc],
                                  file_name=filename_validation,
                                  figure_size=(7, 7))

    # summary
    sim_outcomes.plot_multi_panel(n_rows=2, n_cols=3,
                                  list_plot_info=[ObsInc, ObsIncA, ObsIncB, ObsICU, ObsPercB, ObsPercV],
                                  file_name=filename_summary,
                                  show_subplot_labels=True,
                                  figure_size=(6, 4)
                                  )
