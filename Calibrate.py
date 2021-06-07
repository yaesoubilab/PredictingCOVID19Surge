import _model.COVIDModel as M
import apace.Calibration as calib
from SimulateMany import simulate
from _model.COVIDSettings import COVIDSettings


N_OF_CALIB_SIMS = 500    # total number of trajectories to simulate as part of calibration
N_OF_SIMS = 50   # number of trajectories to simulate using the calibrated model


if __name__ == "__main__":

    # get model settings
    sets = COVIDSettings()
    sets.simulationDuration = 1
    sets.ifCollectTrajsOfCompartments = False
    sets.exportCalibrationTrajs = False

    # calibrate the model
    calibration = calib.CalibrationWithRandomSampling(model_settings=sets)

    calibration.run(
        function_to_populate_model=M.build_covid_model,
        num_of_iterations=N_OF_CALIB_SIMS,
        if_run_in_parallel=True)

    # save calibration results
    calibration.save_results(filename='summaries/calibration_summary.csv')

    # get the seeds and probability weights
    seeds, weights = calib.get_seeds_and_probs('summaries/calibration_summary.csv')

    # simulate the calibrated model
    simulate(n=N_OF_SIMS,
             seeds=seeds,
             weights=weights,
             sample_seeds_by_weights=False)
