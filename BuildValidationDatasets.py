from BuildTrainingDataset import build_and_combine_datasets, WEEKS_IN_FALL, HOSP_OCCU_THRESHOLDS


def build_validation_datasets():

    # build the dataset with smaller survey size
    build_and_combine_datasets(
        type_of_dataset='smaller survey',
        weeks_in_fall=WEEKS_IN_FALL,
        weeks_to_predict=4,
        hosp_occu_thresholds=HOSP_OCCU_THRESHOLDS,
        survey_size_novel_inf=200
    )

    # build the dataset for the base scenario

    # build the dataset with no novel variant

    # build the dataset with no mitigating strategies


if __name__ == '__main__':
    build_validation_datasets()
