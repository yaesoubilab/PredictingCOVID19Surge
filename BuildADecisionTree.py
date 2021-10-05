import pandas as pd

from covid_prediction.model_specs import A, B3, SHORT_FEATURE_NAMES
from covid_prediction.prediction_models import DecisionTree
from definitions import ROOT_DIR


def build_a_decision_tree(feature_names, outcome_name, max_depth=None, ccp_alpha=0.0, fig_filename='tree.png'):

    # read data
    dataset = pd.read_csv(ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/combined_data.csv')

    # create a decision tree
    dt = DecisionTree(df=dataset, feature_names=feature_names, y_name=outcome_name)

    # train the decision tree
    dt.run(test_size=0.2, max_depth=max_depth, ccp_alpha=ccp_alpha)

    # print selected features
    print(dt.selectedFeatures)

    # report the performance of the decision tree on the testing set
    dt.performanceTest.print()

    # plot the decision path
    dt.plot_decision_path(file_name=fig_filename, simple=True, class_names=['Yes', 'No'],
                          impurity=True, proportion=True, label='all', precision=2,
                          shorten_feature_names=SHORT_FEATURE_NAMES)


if __name__ == '__main__':

    for model in (A, B3):
        build_a_decision_tree(feature_names=model.features,
                              outcome_name='If threshold passed (0:Yes)',
                              ccp_alpha=0.01,
                              fig_filename=ROOT_DIR+'/outputs/figures/trees/model-{}.png'.format(model.name))

