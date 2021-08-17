
class ModelDef:
    def __init__(self, name, features, list_num_of_features_wanted, list_num_of_neurons):
        self.name = name
        self.features = features
        self.listNumOfFeaturesWanted = list_num_of_features_wanted
        self.listNumOfNeurons = list_num_of_neurons


FULL = ModelDef(name='Full',
                features=None,
                list_num_of_features_wanted=[10, 20],
                list_num_of_neurons=[10, 20])

A = ModelDef(name='A',
             features=[
                 'Obs: New hospitalization rate',
                 'Obs: Cumulative vaccination rate',
                 'Obs: Cumulative hospitalization rate'],
             list_num_of_features_wanted=[2, 3],
             list_num_of_neurons=[3, 4])

B = ModelDef(name='B',
             features=[],
             list_num_of_features_wanted=[],
             list_num_of_neurons=[3, 4])
