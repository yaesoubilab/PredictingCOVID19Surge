from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree
from covid_prediction.print_features import print_selected_features_dec_trees
from definitions import ROOT_DIR, OUTCOMES_IN_DATASET

MODELS = (A, B3)

ALPHAS = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = False
# MAX_DEPTHS = [3, 4, 5]
# FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


def evaluate(noise_coeff):
    """
    :param noise_coeff: (None or int) if None, the noise model is not added, otherwise, the noise model is
        added with survey size multiplied by add_noise.
    """

    # make prediction at different weeks
    rows = [['Model', 'CV-Score', 'CV-PI', 'CV-Formatted PI', 'Validation-Accuracy']]

    for model in MODELS:

        print("Evaluating model {}.".format(model.name))

        # model zero assumes no noise or bias
        best_spec, final_model_performance = optimize_and_eval_dec_tree(
            model_spec=model,
            outcome_name=OUTCOMES_IN_DATASET[1],
            list_of_max_depths=None,
            list_of_ccp_alphas=ALPHAS,
            feature_selection=None,
            cv_fold=CV_FOLD,
            if_parallel=IF_PARALLEL,
            shorten_feature_names=SHORT_FEATURE_NAMES)

        # store outcomes
        rows.append([model.name,
                     best_spec.meanScore,
                     best_spec.PI,
                     best_spec.formattedMeanPI,
                     final_model_performance.accuracy])

    # print summary of results
    write_csv(rows=rows,
              file_name=ROOT_DIR+'/outputs/prediction_summary/dec_tree/summary.csv')

    # print features by model

    print_selected_features_dec_trees(models=MODELS)
    #
    # # plot
    # plot_performance(noise_coeff=noise_coeff, bias_delay=bias_delay)


if __name__ == '__main__':

    evaluate(noise_coeff=None)
    # evaluate(noise_coeff=1)
    # evaluate(noise_coeff=0.5, bias_delay=4)

