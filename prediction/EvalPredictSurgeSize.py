from SimPy.InOutFunctions import write_csv
from covid_prediction.model_definitions import *
from covid_prediction.optimize_parameters import get_nue_net_best_performance
from definitions import ROOT_DIR

WEEKS = (-12, -8, -4)
CV_FOLD = 10         # num of splits for cross validation
IF_PARALLEL = True
ALPHAS = [0.0001, 0.001]
IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


if __name__ == '__main__':

    # make prediction at different weeks
    rows = [['Week', 'Model', 'R2', 'R2 (PI)']]

    for week in WEEKS:
        for model in (FULL, A):

            print('Evaluating model {} at week {}.'.format(model.name, week))

            # find the best model specification
            best_spec = get_nue_net_best_performance(
                week=week, model_definition=model,
                list_of_alphas=ALPHAS, feature_selection=FEATURE_SELECTION,
                if_standardize=IF_STANDARDIZED, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

            # store outcomes
            rows.append([week, model.name, best_spec.meanScore, best_spec.PI])

    write_csv(rows=rows, file_name=ROOT_DIR+'/outputs/prediction_summary/summary.csv')
