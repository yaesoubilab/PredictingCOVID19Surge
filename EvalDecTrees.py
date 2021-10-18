import numpy as np

from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree
from definitions import ROOT_DIR, HOSP_OCCU_THRESHOLDS, SCENARIOS, DIGITS, CV_FOLD

MODELS = (A, B3)

ALPHAS = np.arange(0, 0.1, 0.005) # [0, 0.01, 0.02, 0.03, 0.04, 0.05]
IF_PARALLEL = False


def evaluate():

    # make prediction at different weeks
    rows = [['Model', 'Threshold', 'CV-Formatted PI']] # 'CV-Score', 'CV-PI',
    for key, value in SCENARIOS.items():
        rows[0].extend(['Acc-' + value, 'Sen-' + value, 'Spe-' + value])

    for model in MODELS:

        print("Evaluating model {}.".format(model.name))

        best_spec_and_validation_performance = optimize_and_eval_dec_tree(
            model_spec=model,
            hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
            list_of_ccp_alphas=ALPHAS,
            cv_fold=CV_FOLD,
            if_parallel=IF_PARALLEL,
            shorten_feature_names=SHORT_FEATURE_NAMES)

        for t in HOSP_OCCU_THRESHOLDS:

            # best specification and performance
            best_spec, performance = best_spec_and_validation_performance[str(t)]

            # store results
            result = [model.name,
                      t,
                      # best_spec.meanScore,
                      # best_spec.PI,
                      best_spec.get_formatted_mean_and_interval(deci=DIGITS)]
            for p in performance:
                result.extend([
                    round(p.accuracy, DIGITS),
                    None if p.sen is None else round(p.sen, DIGITS),
                    None if p.spe is None else round(p.spe, DIGITS)
                ])

            rows.append(result)

    # print summary of results
    write_csv(rows=rows,
              file_name=ROOT_DIR+'/outputs/prediction_summary/dec_tree/summary.csv')


if __name__ == '__main__':

    evaluate()
    # evaluate(noise_coeff=1)
    # evaluate(noise_coeff=0.5, bias_delay=4)

