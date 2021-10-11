import apace.analysis.Trajectories as T
from SimPy.Statistics import SummaryStat
from definitions import FEASIBILITY_PERIOD, SIM_DURATION, ROOT_DIR


def report_traj_summary(hosp_occ_thresholds):
    """
    :param hosp_occ_thresholds: (list) rates (per 100,000 population) for threshold of hospital occupancy
    :return:
    """

    sim_outcomes = T.SimOutcomeTrajectories(
        csv_directory=ROOT_DIR+'/outputs/trajectories')

    # report % trajectories with hospital occupancy passing certain threshold
    hosp_occ_all_trajs = sim_outcomes.dictOfSimOutcomeTrajectories['Obs: Hospital occupancy rate']. \
        get_obss_over_time_period(period_time_index=[FEASIBILITY_PERIOD*52, SIM_DURATION*52])

    new_hosp_all_trajs = sim_outcomes.dictOfSimOutcomeTrajectories['Obs: New hospitalization rate']. \
        get_obss_over_time_period(period_time_index=[FEASIBILITY_PERIOD*52, SIM_DURATION*52])

    list_n_with_hosp_occ_passing_threshold = [0] * len(hosp_occ_thresholds)
    list_of_max_hosp_occ = []
    new_hosp_all_trajs = []

    for hosp_occ_for_this_traj in hosp_occ_all_trajs:
        # find the maximum:
        max_hosp_occ = max(hosp_occ_for_this_traj)
        list_of_max_hosp_occ.append(max_hosp_occ)
        for i, t in enumerate(hosp_occ_thresholds):
            if max_hosp_occ > t/100000:
                list_n_with_hosp_occ_passing_threshold[i] += 1

    for new_hosp_for_this_traj in new_hosp_all_trajs:
        # find the maximum:
        max_new_hosp = max(new_hosp_for_this_traj)
        new_hosp_all_trajs.append(max_new_hosp)

    # report
    for i, t in enumerate(hosp_occ_thresholds):
        print('% trajectories passing the hospital occupancy threshold {}: '.format(t),
              round(100 * list_n_with_hosp_occ_passing_threshold[i] / len(hosp_occ_all_trajs), 1))

    sum_stat_max_hosp_occ = SummaryStat(data=list_of_max_hosp_occ)
    print('Max rate of hospital occupancy (mean, PI): ',
          sum_stat_max_hosp_occ.get_formatted_mean_and_interval(
              interval_type='p', deci=1, multiplier=100000))


if __name__ == '__main__':

    report_traj_summary(hosp_occ_thresholds=[10, 20, 30])
