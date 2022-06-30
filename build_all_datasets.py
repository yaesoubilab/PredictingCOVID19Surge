from covid_prediction.build_training_datasets import build_training_dataset
from covid_prediction.build_validation_datasets import build_validation_datasets

"""
This scrip builds the datasets needed to train and validate the decision rules
for prediction over the next 4 and 8 weeks.  
"""

if __name__ == "__main__":

    for weeks_to_predict in (4, 8):

        build_training_dataset(weeks_to_predict=weeks_to_predict)
        build_validation_datasets(weeks_to_predict=weeks_to_predict)
