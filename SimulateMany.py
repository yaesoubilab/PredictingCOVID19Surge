import apace.Calibration as calib
from covid_model import COVIDModel as M
from covid_model.COVIDSettings import COVIDSettings
from covid_visualization.PlotTrajs import plot
from apace.MultiEpidemics import MultiEpidemics

N = 8
IF_PARALLEL = True
USE_CALIBRATED_MODEL = False


def simulate(n=25, calibrated=True, seeds=None, weights=None, sample_seeds_by_weights=False):

    # get model settings
    sets = COVIDSettings()

    # build multiple epidemics
    multi_model = MultiEpidemics(model_settings=sets)

    if calibrated:
        # get the seeds and probability weights
        seeds, weights = calib.get_seeds_and_probs('output/summaries/calibration_summary.csv')

    multi_model.simulate(function_to_populate_model=M.build_covid_model,
                         n=n,
                         seeds=seeds,
                         weights=weights,
                         sample_seeds_by_weights=sample_seeds_by_weights,
                         if_run_in_parallel=IF_PARALLEL)

    # save ids, seeds, runtime,
    multi_model.save_summary()

    # get summary statistics of runtime,
    multi_model.print_summary_stats()

    # plot trajectories
    plot(prev_multiplier=52,  # to show weeks on the x-axis of prevalence data
         incd_multiplier=sets.simulationOutputPeriod * 52, # to show weeks on the x-axis of incidence data
         obs_incd_multiplier=sets.observationPeriod*52
         )


if __name__ == "__main__":

    simulate(n=N, calibrated=USE_CALIBRATED_MODEL)

