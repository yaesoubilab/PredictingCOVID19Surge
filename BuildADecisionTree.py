import pandas as pd

from covid_prediction.model_specs import A, SHORT_FEATURE_NAMES
from covid_prediction.prediction_models import DecisionTree
from definitions import ROOT_DIR


def build_a_decision_tree(feature_names, outcome_name, max_depth, fig_filename):

    # read data
    dataset = pd.read_csv(ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/combined_data.csv')

    # create a decision tree
    dt = DecisionTree(df=dataset, features=feature_names, y_name=outcome_name)

    # train the decision tree
    dt.run(test_size=0.2, criterion="entropy", max_depth=max_depth)

    # report the performance of the decision tree on the testing set
    dt.performanceTest.print()

    # plot the decision path
    dt.plot_decision_path(file_name=fig_filename, simple=True, class_names=['Yes', 'No'],
                          impurity=True, proportion=False, label='all', precision=6,
                          shorten_feature_names=SHORT_FEATURE_NAMES)


if __name__ == '__main__':

    build_a_decision_tree(feature_names=A.features,
                          outcome_name='If hospitalization threshold passed',
                          max_depth=5,
                          fig_filename=ROOT_DIR+'/outputs/figures/trees/model-{}.png'.format(A.name))

