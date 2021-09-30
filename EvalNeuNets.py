from SimPy.InOutFunctions import write_csv
from covid_prediction.model_specs import *
from covid_prediction.optimize_parameters import get_neural_net_best_spec
from covid_prediction.print_features import print_selected_features
from covid_visualization.plot_prediction import plot_performance
from definitions import ROOT_DIR, get_dataset_labels, get_short_outcome

MODELS = (A, B1, B2 ) #, B2, B3, B4, C1, C2)
OUTCOMES = ('Maximum hospitalization rate', 'If hospitalization threshold passed')
WEEKS = (-12, -8, -4)

CV_FOLD = 20         # num of splits for cross validation
IF_PARALLEL = True
ALPHAS = [0.001]
IF_STANDARDIZED = True
FEATURE_SELECTION = 'pi'  # could be 'rfe', 'lasso', or 'pi'


def evaluate(noise_coeff, bias_delay=None):
    """
    :param noise_coeff: (None or int) if None, the noise model is not added, otherwise, the noise model is
        added with survey size multiplied by add_noise.
    :param bias_delay: (None or int): delay (in weeks) of observing the true value
    """

    for outcome in OUTCOMES:
        short_outcome = get_short_outcome(outcome)
        # make prediction at different weeks
        rows = [['Week', 'Model', 'Score', 'PI', 'Formatted PI']]

        for week in WEEKS:
            for model in MODELS:

                # find the label
                label = get_dataset_labels(
                    week=week, noise_coeff=noise_coeff, bias_delay=bias_delay)

                print("Evaluating model {} with outcome '{}' at {}.".format(
                    model.name, outcome, label))

                # model zero assumes no noise or bias
                if model == Zero:
                    best_spec = get_neural_net_best_spec(
                        outcome_name=outcome,
                        week=week, model_spec=model, noise_coeff=None, bias_delay=None,
                        list_of_alphas=ALPHAS, feature_selection=FEATURE_SELECTION,
                        if_standardize=IF_STANDARDIZED, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

                else:
                    best_spec = get_neural_net_best_spec(
                        outcome_name=outcome,
                        week=week, model_spec=model, noise_coeff=noise_coeff, bias_delay=bias_delay,
                        list_of_alphas=ALPHAS, feature_selection=FEATURE_SELECTION,
                        if_standardize=IF_STANDARDIZED, cv_fold=CV_FOLD, if_parallel=IF_PARALLEL)

                # store outcomes
                rows.append([week, model.name, best_spec.meanScore, best_spec.PI, best_spec.formattedMeanPI])

            # print summary of results
            label = get_dataset_labels(week=None, noise_coeff=noise_coeff, bias_delay=bias_delay)
            write_csv(rows=rows,
                      file_name=ROOT_DIR+'/outputs/prediction_summary/neu_net/predicting {}-summary{}.csv'
                      .format(short_outcome, label))

        # print features by model
        print_selected_features(short_outcome=short_outcome,
                                weeks=WEEKS, models=MODELS,
                                noise_coeff=noise_coeff, bias_delay=bias_delay)

    # plot
    plot_performance(noise_coeff=noise_coeff, bias_delay=bias_delay)


if __name__ == '__main__':

    evaluate(noise_coeff=None)
    # evaluate(noise_coeff=1)
    # evaluate(noise_coeff=0.5, bias_delay=4)

