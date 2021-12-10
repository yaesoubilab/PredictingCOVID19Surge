import numpy as np

from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree, SummaryOfTreePerformance
from definitions import ROOT_DIR, HOSP_OCCU_THRESHOLDS, DIGITS, CV_FOLD, WEEKS_TO_PREDICT

MODELS = (A, B)

ALPHAS = np.arange(0.001, 0.020, 0.0005) # [0, 0.01, 0.02, 0.03, 0.04, 0.05]
IF_PARALLEL = True

# a pruner tree will be selected if it's accuracy is less that the accuracy of the optimal
# tree by this amount
ERROR_TOLERANCE = 0.01


def evaluate():

    # summary builder
    summary = SummaryOfTreePerformance()

    for model in MODELS:

        print("Evaluating model {}.".format(model.name))

        best_spec_and_validation_performance = optimize_and_eval_dec_tree(
            model_spec=model,
            weeks_to_predict=WEEKS_TO_PREDICT,
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
        file_name=ROOT_DIR + '/outputs/prediction_summary_{}_weeks/dec_tree/summary.csv'.format(WEEKS_TO_PREDICT))


if __name__ == '__main__':

    evaluate()

