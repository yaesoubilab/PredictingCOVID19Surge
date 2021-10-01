from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import get_dec_tree_best_spec
from covid_prediction.print_features import print_selected_features_neu_nets
from definitions import ROOT_DIR

MODELS = (A, B3)
OUTCOMES = 'If hospitalization threshold passed'

CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = False
MAX_DEPTHS = [3, 4, 5]
FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


def evaluate(noise_coeff):
    """
    :param noise_coeff: (None or int) if None, the noise model is not added, otherwise, the noise model is
        added with survey size multiplied by add_noise.
    """

    # make prediction at different weeks
    rows = [['Model', 'Score', 'PI', 'Formatted PI']]

    for model in MODELS:

        print("Evaluating model {}.".format(model.name))

        # model zero assumes no noise or bias
        best_spec = get_dec_tree_best_spec(
            model_spec=model, list_of_max_depths=MAX_DEPTHS,
            feature_selection=FEATURE_SELECTION, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

        # store outcomes
        rows.append([model.name, best_spec.meanScore, best_spec.PI, best_spec.formattedMeanPI])

        # print summary of results
        write_csv(rows=rows,
                  file_name=ROOT_DIR+'/outputs/prediction_summary/dec_tree/summary.csv')

    # print features by model
    print_selected_features_neu_nets(short_outcome=short_outcome,
                                     weeks=WEEKS, models=MODELS,
                                     noise_coeff=noise_coeff, bias_delay=bias_delay)
    #
    # # plot
    # plot_performance(noise_coeff=noise_coeff, bias_delay=bias_delay)


if __name__ == '__main__':

    evaluate(noise_coeff=None)
    # evaluate(noise_coeff=1)
    # evaluate(noise_coeff=0.5, bias_delay=4)

