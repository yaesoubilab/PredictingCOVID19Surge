import apacepy.calibration as calib
from apacepy.multi_epidemics import MultiEpidemics

from covid_model import model as M
from covid_model.settings import COVIDSettings
from covid_visualization.plot_trajs import plot

N = 100  # N_SIM_TRAINING  # number of simulation
IF_NOVEL_VARIANT = True     # default True
IF_MITIGATION = True        # default True

IF_PARALLEL = True
USE_CALIBRATED_MODEL = True


def simulate(n=25,
             n_to_display=None,
             calibrated=True, seeds=None, weights=None, sample_seeds_by_weights=False,
             novel_variant_will_emerge=True, mitigating_strategies_on=True,
             print_summary_stats=True,
             folder_to_save_plots=None):

    print('Simulating ...')

    # get model settings
    sets = COVIDSettings(
        novel_variant_will_emerge=novel_variant_will_emerge,
        mitigating_strategies_on=mitigating_strategies_on)

    # build multiple epidemics
    multi_model = MultiEpidemics(model_settings=sets)

    if calibrated and seeds is None:
        # get the seeds and probability weights
        seeds = calib.get_seeds_with_non_zero_prob(
            filename='outputs/summary/calibration_summary.csv',
            random_state=0)
        # seeds, weights = calib.get_seeds_and_probs('outputs/summary/calibration_summary.csv')

    multi_model.simulate(function_to_populate_model=M.build_covid_model,
                         n=n,
                         seeds=seeds,
                         weights=weights,
                         sample_seeds_by_weights=sample_seeds_by_weights,
                         if_run_in_parallel=IF_PARALLEL)

    # save ids, seeds, runtime,
    if print_summary_stats:
        multi_model.save_summary()

    # get summary statistics of runtime,
    if print_summary_stats:
        multi_model.print_summary_stats()

    # plot trajectories
    if n_to_display is not None:
        plot(prev_multiplier=52,  # to show weeks on the x-axis of prevalence data
             incd_multiplier=sets.simulationOutputPeriod * 52,  # to show weeks on the x-axis of incidence data
             obs_incd_multiplier=sets.observationPeriod*52,
             n_random_trajs_to_display=n_to_display,
             save_plots_dir=folder_to_save_plots
             )


if __name__ == "__main__":

    simulate(n=N,
             n_to_display=min(100, N),
             calibrated=USE_CALIBRATED_MODEL,
             novel_variant_will_emerge=IF_NOVEL_VARIANT,
             mitigating_strategies_on=IF_MITIGATION)

