from deampy.InOutFunctions import read_csv_rows, write_csv

from covid_prediction.model_specs import *
from definitions import ROOT_DIR, get_dataset_labels, get_short_outcome

WEEKS = (-12, -8, -4)
MODELS = (A, B1, B2) # (Zero, A, B1, B2, B3, B4, C1, C2)

ORDER_OF_FEATURES = [
    'New hospitalization rate (2-wk average)',
    'New hospitalization rate (4-wk change)',
    'Cumulative hospitalization rate',
    'Cumulative vaccination rate',
    '% of population susceptible',
    '% vaccinated among new hospitalizations (2-wk average)',
    '% vaccinated among new hospitalizations (4-wk change)',
    '% novel variant among new infections (2-wk average)',
    '% novel variant among new infections (4-wk change)',
    '% novel variant among new hospitalizations (2-wk average)',
    '% novel variant among new hospitalizations (4-wk change)',
    '% of new hospitalizations that are vaccinated and due to novel variant (2-wk average)',
    '% of new hospitalizations that are vaccinated and due to novel variant (4-wk change)',
]


def get_all_feature_names():

    # read feature names
    feature_names = read_csv_rows(
        file_name=ROOT_DIR+'/outputs/prediction_datasets/features.csv',
        if_ignore_first_row=False)
    feature_names = [f[0] for f in feature_names]

    # read original feature names and cleaned feature names
    f_names_and_cleaned_names = read_csv_rows(
        file_name=ROOT_DIR + '/covid_prediction/alternative feature names.csv',
        if_ignore_first_row=False)

    # make a dictionary of cleaned name with original name as keys
    dict_cleaned_feature_names = {}
    for row in f_names_and_cleaned_names:
        dict_cleaned_feature_names[row[0]] = row[1]

    return feature_names, dict_cleaned_feature_names


def print_selected_features_neu_nets(short_outcome, weeks, models, noise_coeff, bias_delay):

    # get the names of all features
    feature_names, dict_cleaned_feature_names = get_all_feature_names()

    # make a dictionary with all (cleaned) feature names as key
    dic_cleaned_features_and_selection_results = {}
    for r, v in dict_cleaned_feature_names.items():
        dic_cleaned_features_and_selection_results[v] = [None] * len(weeks) * len(models)

    # read all files with selected features
    i = 0
    col_names = ['features']
    for model in models:
        for week in weeks:

            if model == Zero: # no error or bias for the Zero model
                label = get_dataset_labels(week=week, survey_size=None, bias_delay=None)
            else:
                label = get_dataset_labels(week=week, survey_size=noise_coeff, bias_delay=bias_delay)
            filename = '/outputs/prediction_summary/neu_net/features/features-predicting {}-{}-{}.csv'.format(
                short_outcome, model.name, label)
            col_name = '{} weeks until peak | model {}'.format(-week, model.name)
            col_names.append(col_name)
            # read file
            selected_features = read_csv_rows(file_name=ROOT_DIR + filename,
                                              if_ignore_first_row=False)
            selected_features = [f[0] for f in selected_features]

            # in the list of all features, mark the features selected in this model
            if model.features is not None:
                for f in model.features:
                    cleaned_f = dict_cleaned_feature_names[f]
                    if f in selected_features:
                        dic_cleaned_features_and_selection_results[cleaned_f][i] = '+'
                    else:
                        dic_cleaned_features_and_selection_results[cleaned_f][i] = '-'
            else:
                for f in feature_names:
                    cleaned_f = dict_cleaned_feature_names[f]
                    if f in selected_features:
                        dic_cleaned_features_and_selection_results[cleaned_f][i] = '+'
                    else:
                        dic_cleaned_features_and_selection_results[cleaned_f][i] = '-'

            i += 1

    # put results into rows (dictionary with feature names as keys)
    dict_result = {'features': col_names[1:]}
    for f, v in dic_cleaned_features_and_selection_results.items():
        dict_result[f] = v

    # order rows to be consistent with the table in the manuscript
    result = [col_names]
    for f in ORDER_OF_FEATURES:
        row = [f]
        row.extend(dict_result[f])
        result.append(row)

    label = get_dataset_labels(week=None, survey_size=noise_coeff, bias_delay=bias_delay)
    write_csv(rows=result,
              file_name=ROOT_DIR+'/outputs/prediction_summary/neu_net/selected features for predicting {}-{}.csv'.format(
                  short_outcome, label))


def print_selected_features_dec_trees(models):
    print('Needs to be implemented.')


if __name__ == '__main__':

    for outcome in ('Maximum hospitalization rate', 'If hospitalization threshold passed'):

        short_outcome = get_short_outcome(outcome=outcome)

        # without noise or delay
        print_selected_features_neu_nets(short_outcome=short_outcome, weeks=WEEKS,
                                         models=MODELS, noise_coeff=None, bias_delay=None)
        # with noise
        print_selected_features(short_outcome=short_outcome, weeks=WEEKS,
                                models=MODELS, noise_coeff=1, bias_delay=None)
        # with noise and delay
        print_selected_features(short_outcome=short_outcome, weeks=WEEKS,
                                models=MODELS, noise_coeff=0.5, bias_delay=4)
