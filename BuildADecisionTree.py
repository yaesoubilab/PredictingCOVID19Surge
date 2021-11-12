from covid_prediction.model_specs import A, SHORT_FEATURE_NAMES
from covid_prediction.optimize_parameters import optimize_and_eval_dec_tree

ALPHA = 0.003
MODEL = A # B
THRESHOLD = 10


def build_a_decision_tree(model_spec, hosp_occu_threshold, ccp_alpha=0.0):

    validation_performance = optimize_and_eval_dec_tree(
        model_spec=model_spec,
        hosp_occu_thresholds=[hosp_occu_threshold],
        optimal_ccp_alpha=ccp_alpha,
        shorten_feature_names=SHORT_FEATURE_NAMES)

    for p in validation_performance[str(hosp_occu_threshold)][1]:
        print(p)


if __name__ == '__main__':

    build_a_decision_tree(model_spec=MODEL,
                          hosp_occu_threshold=THRESHOLD,
                          ccp_alpha=ALPHA)

