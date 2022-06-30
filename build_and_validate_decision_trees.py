import numpy as np

from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree, SummaryOfTreePerformance
from definitions import ROOT_DIR, HOSP_OCCU_THRESHOLDS, DIGITS, CV_FOLD

MODELS = (A, B)  # decision trees to evaluate
ALPHAS = np.arange(0.001, 0.020, 0.001)  # values of ccp penalty
IF_PARALLEL = True

# a pruner tree will be selected if it's accuracy is less that the accuracy of the optimal
# tree by this amount
ERROR_TOLERANCE = 0.005


def evaluate(weeks_to_predict):

    # summary builder
    summary = SummaryOfTreePerformance()

    for model in MODELS:

        print("Evaluating model {} for {}-week prediction.".format(model.name, weeks_to_predict))

        best_spec_and_validation_performance = optimize_and_eval_dec_tree(
            model_spec=model,
            weeks_to_predict=weeks_to_predict,
            hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
            list_of_ccp_alphas=ALPHAS,
            error_tolerance=ERROR_TOLERANCE,
            cv_fold=CV_FOLD,
            if_parallel=IF_PARALLEL,
            shorten_feature_names=SHORT_FEATURE_NAMES)

        summary.add(model_name=model.name,
                    hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
                    best_spec_and_validation_performance=best_spec_and_validation_performance,
                    digits=DIGITS)

    # generate the report
    summary.print(
        file_name=ROOT_DIR + '/outputs/prediction_summary_{}_weeks/dec_tree/summary.csv'.format(weeks_to_predict))


if __name__ == '__main__':

    evaluate(weeks_to_predict=4)
    evaluate(weeks_to_predict=8)

