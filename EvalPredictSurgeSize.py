from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import get_neural_net_best_spec
from covid_prediction.print_features import print_selected_features
from covid_visualization.plot_prediction import plot_performance
from definitions import ROOT_DIR


MODELS = (A, B1, B2, B3, B4, C1, C2)

WEEKS = (-12, -8, -4)
CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = True
ALPHAS = [0.001]
IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


if __name__ == '__main__':

    # make prediction at different weeks
    rows = [['Week', 'Model', 'R2', 'error', 'PI']]

    for week in WEEKS:
        for model in MODELS:

            print('Evaluating model {} at week {}.'.format(
                model.name, week))

            # find the best model specification
            best_spec = get_neural_net_best_spec(
                week=week, model_spec=model,
                list_of_alphas=ALPHAS, feature_selection=FEATURE_SELECTION,
                if_standardize=IF_STANDARDIZED, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

            # store outcomes
            rows.append([week, model.name, best_spec.meanScore, best_spec.error, best_spec.PI])

    write_csv(rows=rows, file_name=ROOT_DIR+'/outputs/prediction_summary/summary.csv')

    # plot
    plot_performance()

    # print features by model
    print_selected_features()
