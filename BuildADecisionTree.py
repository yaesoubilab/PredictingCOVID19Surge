from covid_prediction.model_specs import B3, SHORT_FEATURE_NAMES
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree, SummaryOfTreePerformance
from definitions import ROOT_DIR, DIGITS

ALPHA = 0.005
MODEL = B3
THRESHOLD = 15


def build_a_decision_tree(model_spec, hosp_occu_threshold, ccp_alpha=0.0):

    # summary builder
    summary = SummaryOfTreePerformance()

    validation_performance = optimize_and_eval_dec_tree(
        model_spec=model_spec,
        hosp_occu_thresholds=[hosp_occu_threshold],
        optimal_ccp_alpha=ccp_alpha,
        shorten_feature_names=SHORT_FEATURE_NAMES)

    summary.add(model_name=model_spec.name,
                hosp_occu_thresholds=[hosp_occu_threshold],
                best_spec_and_validation_performance=validation_performance,
                digits=DIGITS)

    # generate the report
    summary.print(
        file_name=ROOT_DIR + '/outputs/prediction_summary/dec_tree/summary-{}-{}-{}.csv'.format(
            model_spec.name, hosp_occu_threshold, ccp_alpha))


if __name__ == '__main__':

    build_a_decision_tree(model_spec=MODEL,
                          hosp_occu_threshold=THRESHOLD,
                          ccp_alpha=ALPHA)

