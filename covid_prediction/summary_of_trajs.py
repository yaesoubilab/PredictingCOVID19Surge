import apace.analysis.Trajectories as T
from SimPy.Statistics import SummaryStat
from definitions import FEASIBILITY_PERIOD, SIM_DURATION, ROOT_DIR


def print_stat_summary(name, obss):

    stat = SummaryStat(data=obss)
    print(name,
          stat.get_formatted_mean_and_interval(interval_type='p', deci=1, multiplier=100000))


def get_trajs_over_training_period(sim_trajs, outcome_name):

    assert isinstance(sim_trajs, T.SimOutcomeTrajectories)

    return sim_trajs.dictOfSimOutcomeTrajectories[outcome_name].get_obss_over_time_period(
        period_time_index=[FEASIBILITY_PERIOD*52, SIM_DURATION*52])


def report_traj_summary(hosp_occ_thresholds):
    """
    :param hosp_occ_thresholds: (list) rates (per 100,000 population) for threshold of hospital occupancy
    :return:
    """

    # read trajectories
    sim_trajs = T.SimOutcomeTrajectories(
        csv_directory=ROOT_DIR+'/outputs/trajectories')

    hosp_occ_all_trajs = get_trajs_over_training_period(
        sim_trajs=sim_trajs,
        outcome_name='Obs: Hospital occupancy rate')
    new_hosp_all_trajs = get_trajs_over_training_period(
        sim_trajs=sim_trajs,
        outcome_name='Obs: New hospitalization rate')
    immunity_all_trajs = get_trajs_over_training_period(
        sim_trajs=sim_trajs,
        outcome_name='Obs: Prevalence with immunity from infection')

    # list of desired statistics
    list_n_with_hosp_occ_passing_threshold = [0] * len(hosp_occ_thresholds)
    list_of_max_hosp_occ = []
    list_of_max_new_hosp = []
    min_immunity = float('inf')
    max_immunity = float('-inf')

    # find the maximum of hospital occupancy
    for hosp_occ_for_this_traj in hosp_occ_all_trajs:
        max_hosp_occ = max(hosp_occ_for_this_traj)
        list_of_max_hosp_occ.append(max_hosp_occ)

        # if hospital occupancy surpassed the capacity
        for i, t in enumerate(hosp_occ_thresholds):
            if max_hosp_occ > t/100000:
                list_n_with_hosp_occ_passing_threshold[i] += 1

    # find the maximum of new hospitalizations
    for new_hosp_for_this_traj in new_hosp_all_trajs:
        # find the maximum:
        max_new_hosp = max(new_hosp_for_this_traj)
        list_of_max_new_hosp.append(max_new_hosp)
        
    # find min and max of immunity prevalence
    for immunity_for_this_traj in immunity_all_trajs:
        minimum = min(immunity_for_this_traj)
        maximum = max(immunity_for_this_traj)
        if minimum < min_immunity:
            min_immunity = minimum
        if maximum > max_immunity:
            max_immunity = maximum

    # report % trajectories with hospital occupancy surpassed each threshold
    for i, t in enumerate(hosp_occ_thresholds):
        print('% trajectories passing the hospital occupancy threshold {}: '.format(t),
              round(100 * list_n_with_hosp_occ_passing_threshold[i] / len(hosp_occ_all_trajs), 1))

    # maximum rate of hospital occupancy
    print_stat_summary(name='Max rate of hospital occupancy (mean, PI): ',
                       obss=list_of_max_hosp_occ)
    print_stat_summary(name='Max rate of new hospitalizations (mean, PI): ',
                       obss=list_of_max_new_hosp)
    print('Range of immunity from infection: [{:.1%}, {:.1%}]'.format(min_immunity, max_immunity))
    

if __name__ == '__main__':

    report_traj_summary(hosp_occ_thresholds=[10, 20, 30])
