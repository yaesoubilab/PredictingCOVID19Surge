
class ModelDef:
    def __init__(self, name, features, list_num_of_features_wanted, list_num_of_neurons):
        self.name = name
        self.features = features
        self.listNumOfFeaturesWanted = list_num_of_features_wanted
        self.listNumOfNeurons = list_num_of_neurons


FULL = ModelDef(name='Full',
                features=None,
                list_num_of_features_wanted=[10, 20],
                list_num_of_neurons=[30, 40, 50])

A = ModelDef(name='A',
             features=[
                 'Obs: Incidence rate',
                 'Obs: Incidence rate-ave-2wk',
                 'Obs: Incidence rate-slope-4wk',
                 'Obs: New hospitalization rate',
                 'Obs: New hospitalization rate-ave-2wk',
                 'Obs: New hospitalization rate-slope-4wk',
                 'Obs: % of incidence due to Novel-Unvaccinated',
                 'Obs: % of incidence due to Novel-Unvaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Unvaccinated-slope-4wk',
                 'Obs: % of incidence due to Novel-Vaccinated',
                 'Obs: % of incidence due to Novel-Vaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Vaccinated-slope-4wk',
                 'Obs: % of new hospitalizations due to Novel-Unvaccinated',
                 'Obs: % of new hospitalizations due to Novel-Unvaccinated-ave-2wk',
                 'Obs: % of new hospitalizations due to Novel-Unvaccinated-slope-4wk',
                 'Obs: % of new hospitalizations due to Novel-Vaccinated',
                 'Obs: % of new hospitalizations due to Novel-Vaccinated-ave-2wk',
                 'Obs: % of new hospitalizations due to Novel-Vaccinated-slope-4wk',
                 'Obs: % of incidence with novel variant',
                 'Obs: % of incidence with novel variant-ave-2wk',
                 'Obs: % of incidence with novel variant-slope-4wk',
                 'Obs: % of new hospitalizations with novel variant',
                 'Obs: % of new hospitalizations with novel variant-ave-2wk',
                 'Obs: % of new hospitalizations with novel variant-slope-4wk',
                 'Obs: Prevalence susceptible',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate',
             ],
             list_num_of_features_wanted=[10, 20],
             list_num_of_neurons=[20, 25])

B = ModelDef(name='B',
             features=[],
             list_num_of_features_wanted=[],
             list_num_of_neurons=[3, 4])
