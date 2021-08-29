from SimPy.InOutFunctions import read_csv_rows, write_csv
from covid_prediction.model_specs import Zero, A, B, C, D
from definitions import ROOT_DIR

WEEKS = (-12, -8, -4)
MODELS = (Zero, A, B, C, D)


def print_selected_features():

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

    # make a dictionary with all (cleaned) feature names as key
    dic_cleaned_features_and_selection_results = {}
    for r, v in dict_cleaned_feature_names.items():
        dic_cleaned_features_and_selection_results[v] = [None] * len(WEEKS) * len(MODELS)

    # read all files with selected features
    i = 0
    col_names = ['features']
    for model in MODELS:
        for week in WEEKS:
            filename = '/outputs/prediction_summary/features/NN features-wk {}-model {}.csv'.format(week, model.name)
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

    # print results
    result = [col_names]
    for f, v in dic_cleaned_features_and_selection_results.items():
        row = [f]
        row.extend(v)
        result.append(row)

    write_csv(rows=result, file_name=ROOT_DIR+'/outputs/prediction_summary/selected_features.csv')


if __name__ == '__main__':
    print_selected_features()
