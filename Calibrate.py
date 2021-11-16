import warnings

import apace.Calibration as calib
from SimulateMany import simulate
from covid_model.model import build_covid_model
from covid_model.settings import COVIDSettings
from definitions import N_SIM_CALIBRATION, N_SIM_TRAINING, N_SIM_VALIDATION, CALIB_PERIOD, ROOT_DIR

RUN_IN_PARALLEL = False


if __name__ == "__main__":

    # sys.stdout = open(
    #     ROOT_DIR + '/outputs/summary/calibration.txt', 'w')
    file = open(ROOT_DIR + '/outputs/summary/calibration.txt', 'w')

    # get model settings
    sets = COVIDSettings(if_calibrating=True)
    sets.simulationDuration = CALIB_PERIOD
    sets.ifCollectTrajsOfCompartments = False
    sets.exportCalibrationTrajs = False
    sets.storeParameterValues = False

    # calibrate the model
    calibration = calib.CalibrationWithRandomSampling(
        model_settings=sets, parallelization_approach='few-many', max_tries=100)

    calibration.run(
        function_to_populate_model=build_covid_model,
        num_of_iterations=N_SIM_CALIBRATION,
        if_run_in_parallel=RUN_IN_PARALLEL)

    file.write('Number of calibration processors: {}\n'.format(N_SIM_CALIBRATION))
    file.write('Number of trajectories discarded: {}\n'.format(calibration.nTrajsDiscarded))
    file.write('Calibration duration (seconds): {}\n'.format(round(calibration.runTime, 1)))
    file.write('Number of trajectories with non-zero probability: {}\n'.format(calibration.nTrajsWithNonZeroProb))
    n_trajs_needed = N_SIM_TRAINING+N_SIM_VALIDATION
    if calibration.nTrajsWithNonZeroProb < n_trajs_needed:
        warnings.warn('\nNumber of trajectories with non-zero probability ({}) is less than '
                      'the number of trajectories needed '
                      'for training and validating the predictive models ({}).'
                      '\nIncrease the number of trajectories '
                      'used for calibration which is currently {}.'
                      .format(calibration.nTrajsWithNonZeroProb, n_trajs_needed, N_SIM_CALIBRATION))

    file.close()

    # save calibration results
    calibration.save_results(filename='outputs/summary/calibration_summary.csv')

    # simulate the calibrated model
    simulate(n=N_SIM_TRAINING,
             n_to_display=min(100, N_SIM_TRAINING),
             calibrated=True,
             sample_seeds_by_weights=False)


