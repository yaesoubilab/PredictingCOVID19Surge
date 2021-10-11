import numpy as np

from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree
from covid_prediction.print_features import print_selected_features_dec_trees
from definitions import ROOT_DIR, get_outcome_label, N_NOVEL_INCD

MODELS = (A, B3)

ALPHAS = np.arange(0, 0.1, 0.005) # [0, 0.01, 0.02, 0.03, 0.04, 0.05]
CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = False


def evaluate(hosp_occu_threshold, survey_size_novel_inf):
    """
    :param hosp_occu_threshold: (float) threshold for hospital occupancy (per 100,000 population)
    :param survey_size_novel_inf: (int) survey size of novel infection surveillance
    """

    # make prediction at different weeks
    rows = [['Model', 'CV-Score', 'CV-PI', 'CV-Formatted PI', 'Validation-Accuracy']]

    for model in MODELS:

        print("Evaluating model {}.".format(model.name))

        # model zero assumes no noise or bias
        best_spec, final_model_performance = optimize_and_eval_dec_tree(
            model_spec=model,
            survey_size_novel_inf=survey_size_novel_inf,
            outcome_name=get_outcome_label(threshold=hosp_occu_threshold),
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

    evaluate(hosp_occu_threshold=10, survey_size_novel_inf=N_NOVEL_INCD)
    # evaluate(noise_coeff=1)
    # evaluate(noise_coeff=0.5, bias_delay=4)

