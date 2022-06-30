from apacepy.epidemic import EpiModel

from covid_model import model as M
from covid_model.settings import COVIDSettings
from covid_visualization.plot_trajs import plot


def simulate(seed):

    # get model settings
    sets = COVIDSettings(novel_variant_will_emerge=True, if_calibrating=True)

    # make an (empty) epidemic model
    model = EpiModel(id=1, settings=sets)
    # populate the SIR model
    M.build_covid_model(model)

    # simulate
    model.simulate(seed=seed)
    # print trajectories
    model.export_trajectories(delete_existing_files=True)

    # print discounted outcomes
    print(model.get_total_discounted_cost_and_health())

    # plot trajectories
    plot(prev_multiplier=52,  # to show weeks on the x-axis of prevalence data
         incd_multiplier=sets.simulationOutputPeriod*52,  # to show weeks on the x-axis of incidence data
         obs_incd_multiplier=sets.observationPeriod*52)


if __name__ == "__main__":
    simulate(seed=1282240807)
