from SimPy.InOutFunctions import read_csv_rows
from definitions import ROOT_DIR

# read feature names
rows = read_csv_rows(file_name=ROOT_DIR+'/outputs/prediction_datasets/features.csv',
                     if_ignore_first_row=False)

# make a dictionary with all features as key
dic_of_all_features = {}
for r in rows:
    dic_of_all_features[r] = []


