import apace.analysis.Trajectories as T
from SimPy.Statistics import SummaryStat
from definitions import FEASIBILITY_PERIOD, SIM_DURATION


def report_traj_summary(hosp_occ_threshold):
    """
    :param hosp_occ_threshold: rate (per 100,000 population) for threshold of hospital occupancy
    :return:
    """

    sim_outcomes = T.SimOutcomeTrajectories(csv_directory='outputs/trajectories')

    # report % trajectories with hospital occupancy passing certain threshold
    hosp_occ_all_trajs = sim_outcomes.dictOfSimOutcomeTrajectories['Hospital occupancy rate']. \
        get_obss_over_time_period(period_time_index=[FEASIBILITY_PERIOD*52, SIM_DURATION*52])

    n_with_hosp_occ_passing_threshold = 0
    list_of_max_hosp_occ = []
    for hosp_occ_for_this_traj in hosp_occ_all_trajs:

        # find the maximum:
        max_hosp_occ = max(hosp_occ_for_this_traj)

        list_of_max_hosp_occ.append(max_hosp_occ)
        if max_hosp_occ > hosp_occ_threshold/100000:
            n_with_hosp_occ_passing_threshold += 1

    # report
    print('Percentage of trajectories passing the hospital occupancy threshold: ',
          round(100 * n_with_hosp_occ_passing_threshold / len(hosp_occ_all_trajs), 1))

    sum_stat_max_hosp_occ = SummaryStat(data=list_of_max_hosp_occ)
    print('Max rate of hospital occupancy (mean, PI): ',
          sum_stat_max_hosp_occ.get_formatted_mean_and_interval(
              interval_type='p', deci=1, multiplier=100000))


if __name__ == '__main__':

    report_traj_summary(hosp_occ_threshold=10)
