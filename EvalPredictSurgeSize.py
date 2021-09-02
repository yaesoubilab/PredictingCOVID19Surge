from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import get_neural_net_best_spec
from covid_prediction.print_features import print_selected_features
from covid_visualization.plot_prediction import plot_performance
from definitions import ROOT_DIR

MODELS = (Zero, B1, B3) #, B1, B2, B3, B4, C1, C2)
WEEKS = (-12, -8) #, -8, -4)

CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = True
ALPHAS = [0.001]
IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


def evaluate(noise):
    """
    :param noise: None, 1, or 2 (1 or 2 to multiply the survey size by)
    """

    # make prediction at different weeks
    rows = [['Week', 'Model', 'R2', 'error', 'PI']]

    for week in WEEKS:
        for model in MODELS:

            # find the label
            if noise is None:
                label = 'wk {}'.format(week)
            else:
                label = 'wk {} with noise {}'.format(week, noise)
            print('Evaluating model {} at {}.'.format(
                model.name, label))

            best_spec = get_neural_net_best_spec(
                week=week, model_spec=model, noise=noise,
                list_of_alphas=ALPHAS, feature_selection=FEATURE_SELECTION,
                if_standardize=IF_STANDARDIZED, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

            # store outcomes
            rows.append([week, model.name, best_spec.meanScore, best_spec.error, best_spec.PI])

    # print summary of results
    if noise is None:
        label = ''
    else:
        label = '-with noise {}'.format(noise)
    write_csv(rows=rows, file_name=ROOT_DIR+'/outputs/prediction_summary/summary{}.csv'.format(label))

    # plot
    plot_performance(noise=noise)

    # print features by model
    print_selected_features(noise=noise, weeks=WEEKS, models=MODELS)


if __name__ == '__main__':

    evaluate(noise=None)
    evaluate(noise=1)
