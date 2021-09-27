import pandas as pd

from definitions import ROOT_DIR


def build_a_decision_tree(weeks_in_fall):

    # merge all datasets
    dataframes = []
    for w in weeks_in_fall:
        dataframes.append(pd.read_csv(
            ROOT_DIR + '/outputs/prediction_datasets/week_into_fall/data-wk {}.csv'.format(w)))

    dataset = pd.concat(dataframes)

        

if __name__ == '__main__':
    build_a_decision_tree(weeks_in_fall=[8, 16, 24, 32])
