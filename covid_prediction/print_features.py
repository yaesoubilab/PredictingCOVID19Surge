from SimPy.InOutFunctions import read_csv_rows, write_csv
from covid_prediction.model_specs import Zero, A, B, C, D
from definitions import ROOT_DIR

WEEKS = (-12, -8, -4)
MODELS = (Zero, A, B, C, D)

# read feature names
rows = read_csv_rows(file_name=ROOT_DIR+'/outputs/prediction_datasets/features.csv',
                     if_ignore_first_row=False)

# make a dictionary with all features as key
dic_of_all_features = {}
for r in rows:
    dic_of_all_features[r[0]] = [None] * len(WEEKS) * len(MODELS)


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
                if f in selected_features:
                    dic_of_all_features[f][i] = '+'
                else:
                    dic_of_all_features[f][i] = '-'
        else:
            for f in dic_of_all_features:
                if f in selected_features:
                    dic_of_all_features[f][i] = '+'
                else:
                    dic_of_all_features[f][i] = '-'

        i += 1

# print results
rows = [col_names]
for f, v in dic_of_all_features.items():
    row = [f]
    row.extend(v)
    rows.append(row)

write_csv(rows=rows, file_name=ROOT_DIR+'/outputs/prediction_summary/selected_features.csv')
