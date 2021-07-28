import apace.Calibration as calib
from SimulateMany import simulate
from covid_model.model import build_covid_model
from covid_model.settings import COVIDSettings

N_OF_CALIB_SIMS = 2000    # total number of trajectories to simulate as part of calibration
N_OF_SIMS = 200   # number of trajectories to simulate using the calibrated model
RUN_IN_PARALLEL = True


if __name__ == "__main__":

    # get model settings
    sets = COVIDSettings(if_calibrating=True)
    sets.simulationDuration = 78*7/364  # 78 weeks
    sets.ifCollectTrajsOfCompartments = False
    sets.exportCalibrationTrajs = False

    # calibrate the model
    calibration = calib.CalibrationWithRandomSampling(model_settings=sets)

    calibration.run(
        function_to_populate_model=build_covid_model,
        num_of_iterations=N_OF_CALIB_SIMS,
        if_run_in_parallel=RUN_IN_PARALLEL)

    # save calibration results
    calibration.save_results(filename='outputs/summary/calibration_summary.csv')

    # simulate the calibrated model
    simulate(n=N_OF_SIMS,
             calibrated=True,
             sample_seeds_by_weights=False)
