import covid_prediction.model_specs as models
from covid_prediction.model_specs import SHORT_FEATURE_NAMES
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree, SummaryOfTreePerformance
from definitions import ROOT_DIR, DIGITS

"""
To create and validate a decision tree with an specific value for the ccp penalty. 
The decision rule will be stored under outputs/figures/trees_?_weeks and the 
performance of the decision rule will be stored under outputs/prediction_summary_?_weeks.
"""

CCP_ALPHA = 0.002   # ccp penalty value
MODEL = models.B    # decision tree A vs B
HOSP_OCCUPANCY_THRESHOLD = 10   # the specified hospital occupancy threshold
WEEKS_TO_PREDICT = 8    # weeks to predict


def build_a_decision_tree(model_spec, weeks_to_predict, hosp_occu_threshold, ccp_alpha=0.0):

    # summary builder
    summary = SummaryOfTreePerformance()

    validation_performance = optimize_and_eval_dec_tree(
        model_spec=model_spec,
        weeks_to_predict=weeks_to_predict,
        hosp_occu_thresholds=[hosp_occu_threshold],
        optimal_ccp_alpha=ccp_alpha,
        shorten_feature_names=SHORT_FEATURE_NAMES)

    summary.add(model_name=model_spec.name,
                hosp_occu_thresholds=[hosp_occu_threshold],
                best_spec_and_validation_performance=validation_performance,
                digits=DIGITS)

    # generate the report
    summary.print(
        file_name=ROOT_DIR + '/outputs/prediction_summary_{}_weeks/dec_tree/summary-{}-{}-{}.csv'.format(
            weeks_to_predict, model_spec.name, hosp_occu_threshold, ccp_alpha))


if __name__ == '__main__':

    build_a_decision_tree(model_spec=MODEL,
                          weeks_to_predict=WEEKS_TO_PREDICT,
                          hosp_occu_threshold=HOSP_OCCUPANCY_THRESHOLD,
                          ccp_alpha=CCP_ALPHA)

