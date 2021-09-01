
class ModelSpec:
    # specifications of predictive models based on the features included,
    # number of features wanted, and number of neurons
    def __init__(self, name, features, list_num_of_features_wanted, list_num_of_neurons):
        self.name = name
        self.features = features
        self.listNumOfFeaturesWanted = list_num_of_features_wanted
        self.listNumOfNeurons = list_num_of_neurons


Zero = ModelSpec(name='Zero',
                 features=None,  # this will include all features in the dataset
                 list_num_of_features_wanted=[15, 20, 25, 30, 35],
                 list_num_of_neurons=[40, 50, 60])

A = ModelSpec(name='A',
              features=[
                 # 'Obs: New hospitalization rate',
                 'Obs: New hospitalization rate-ave-2wk',
                 'Obs: New hospitalization rate-slope-4wk',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate',
              ],
              list_num_of_features_wanted=[4],
              list_num_of_neurons=[5, 7])

B = ModelSpec(name='B',
              features=[
                 # 'Obs: New hospitalization rate',
                 'Obs: New hospitalization rate-ave-2wk',
                 'Obs: New hospitalization rate-slope-4wk',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate',
                 # 'Obs: % of incidence with novel variant',
                 'Obs: % of incidence with novel variant-ave-2wk',
                 'Obs: % of incidence with novel variant-slope-4wk',
              ],
              list_num_of_features_wanted=[5, 6],
              list_num_of_neurons=[7, 8, 9])

C = ModelSpec(name='C',
              features=[
                 # 'Obs: New hospitalization rate',
                 'Obs: New hospitalization rate-ave-2wk',
                 'Obs: New hospitalization rate-slope-4wk',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate',
                 # 'Obs: % of incidence due to Novel-Unvaccinated',
                 'Obs: % of incidence due to Novel-Unvaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Unvaccinated-slope-4wk',
                 # 'Obs: % of incidence due to Novel-Vaccinated',
                 'Obs: % of incidence due to Novel-Vaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Vaccinated-slope-4wk',
              ],
              list_num_of_features_wanted=[6, 7, 8],
              list_num_of_neurons=[8, 9, 10])

D = ModelSpec(name='D',
              features=[
                 # 'Obs: New hospitalization rate',
                 'Obs: New hospitalization rate-ave-2wk',
                 'Obs: New hospitalization rate-slope-4wk',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate',
                 # 'Obs: % of incidence due to Novel-Unvaccinated',
                 'Obs: % of incidence due to Novel-Unvaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Unvaccinated-slope-4wk',
                 # 'Obs: % of incidence due to Novel-Vaccinated',
                 'Obs: % of incidence due to Novel-Vaccinated-ave-2wk',
                 'Obs: % of incidence due to Novel-Vaccinated-slope-4wk',
                 'Obs: Prevalence susceptible',
              ],
              list_num_of_features_wanted=[7, 8, 9],
              list_num_of_neurons=[9, 10, 11])

