
class ModelSpec:
    """
    specifications of predictive models based on the features included,
    number of features wanted, and number of neurons
    """

    def __init__(self, name, features, list_num_of_features_wanted, list_num_of_neurons):
        self.name = name
        self.features = features
        self.listNumOfFeaturesWanted = list_num_of_features_wanted
        self.listNumOfNeurons = list_num_of_neurons


Zero = ModelSpec(name='O',
                 features=[
                     # 'Obs: Incidence rate',
                     # 'Obs: Incidence rate-ave-2wk',
                     # 'Obs: Incidence rate-slope-4wk',
                     'Obs: New hospitalization rate',
                     'Obs: New hospitalization rate-ave-2wk',
                     'Obs: New hospitalization rate-slope-4wk',
                     'Obs: % of new hospitalizations that are vaccinated',
                     'Obs: % of new hospitalizations that are vaccinated-ave-2wk',
                     'Obs: % of new hospitalizations that are vaccinated-slope-4wk',
                     'Obs: % of incidence due to novel variant',
                     'Obs: % of incidence due to novel variant-ave-2wk',
                     'Obs: % of incidence due to novel variant-slope-4wk',
                     'Obs: % of new hospitalizations due to novel variant',
                     'Obs: % of new hospitalizations due to novel variant-ave-2wk',
                     'Obs: % of new hospitalizations due to novel variant-slope-4wk',
                     'Obs: % of new hospitalizations due to Novel-V',
                     'Obs: % of new hospitalizations due to Novel-V-ave-2wk',
                     'Obs: % of new hospitalizations due to Novel-V-slope-4wk',
                     'Obs: Cumulative hospitalization rate',
                     'Obs: Cumulative vaccination rate',
                     'Obs: Prevalence susceptible',
                     'R0',
                     'Ratio transmissibility by profile-1',
                     'Ratio transmissibility by profile-2',
                     'Duration of infectiousness-dominant',
                     'Ratio of infectiousness duration by profile-1',
                     'Ratio of infectiousness duration by profile-2',
                     'Prob Hosp for 18-29',
                     'Relative prob hosp by age-0',
                     'Relative prob hosp by age-1',
                     'Relative prob hosp by age-2',
                     'Relative prob hosp by age-4',
                     'Relative prob hosp by age-5',
                     'Relative prob hosp by age-6',
                     'Relative prob hosp by age-7',
                     'Ratio of hospitalization probability by profile-1',
                     'Ratio of hospitalization probability by profile-2',
                     'Duration of R-0',
                     'Duration of R-1',
                     'Duration of R-2',
                     'Vaccine effectiveness against infection-0',
                     'Vaccine effectiveness against infection-1',
                     'Duration of vaccine immunity',
                     'Prob novel strain params-0',
                     'Prob novel strain params-1',
                     'Prob novel strain params-2',
                     'PD Y1 thresholds-0',
                     'PD Y1 thresholds-1',
                     'Change in contacts - PD Y1',
                 ],
                 list_num_of_features_wanted=[20, 25, 30, 35, 40],
                 list_num_of_neurons=[25, 35, 45])

A = ModelSpec(name='A',
              features=[
                  'Obs: New hospitalization rate-ave-2wk',
                  'Obs: New hospitalization rate-slope-4wk',
                  'Obs: Cumulative hospitalization rate',
                  'Obs: Cumulative vaccination rate'
              ],
              list_num_of_features_wanted=[4],
              list_num_of_neurons=[5, 7])

B1 = ModelSpec(name='B1',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: Prevalence susceptible'
               ],
               list_num_of_features_wanted=[4, 5],
               list_num_of_neurons=[6, 7])

B2 = ModelSpec(name='B2',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: % of new hospitalizations that are vaccinated-ave-2wk',
                   'Obs: % of new hospitalizations that are vaccinated-slope-4wk',
               ],
               list_num_of_features_wanted=[5, 6],
               list_num_of_neurons=[7, 8])

B3 = ModelSpec(name='B3',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: % of incidence due to novel variant-ave-2wk',
                   'Obs: % of incidence due to novel variant-slope-4wk'
               ],
               list_num_of_features_wanted=[5, 6],
               list_num_of_neurons=[7, 8])

B4 = ModelSpec(name='B4',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: % of new hospitalizations due to novel variant-ave-2wk',
                   'Obs: % of new hospitalizations due to novel variant-slope-4wk'
               ],
               list_num_of_features_wanted=[5, 6],
               list_num_of_neurons=[7, 8])

C1 = ModelSpec(name='C1',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: % of incidence due to novel variant-ave-2wk',
                   'Obs: % of incidence due to novel variant-slope-4wk',
                   'Obs: % of new hospitalizations due to Novel-V-ave-2wk',
                   'Obs: % of new hospitalizations due to Novel-V-slope-4wk',
               ],
               list_num_of_features_wanted=[6, 7, 8],
               list_num_of_neurons=[8, 9, 10])

C2 = ModelSpec(name='C2',
               features=[
                   'Obs: New hospitalization rate-ave-2wk',
                   'Obs: New hospitalization rate-slope-4wk',
                   'Obs: Cumulative hospitalization rate',
                   'Obs: Cumulative vaccination rate',
                   'Obs: % of new hospitalizations due to novel variant-ave-2wk',
                   'Obs: % of new hospitalizations due to novel variant-slope-4wk',
                   'Obs: % of new hospitalizations due to Novel-V-ave-2wk',
                   'Obs: % of new hospitalizations due to Novel-V-slope-4wk',
                ],
               list_num_of_features_wanted=[6, 7, 8],
               list_num_of_neurons=[8, 9, 10])

