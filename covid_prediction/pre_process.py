from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from SimPy.InOutFunctions import delete_file
from covid_prediction.feature_engineering import *
from definitions import ROOT_DIR, get_dataset_labels


def standardize(x):
    return StandardScaler().fit_transform(x)


class PreProcessor:
    """ class to perform pre-processing steps """
    def __init__(self, df, feature_names, y_name):
        """
        :param df: (panda DataFrame)
        :param feature_names: (list) of feature names to be included
        :param y_name: (string) name of the outcome
        """

        self._df = df
        self._featureNames = feature_names
        self._yName = y_name
        self._X = np.asarray(self._df[self._featureNames])
        self._y = np.asarray(self._df[self._yName])

        # x, df, features after preprocessing
        # for now, we set them to the default values
        self.df = self._df
        self.X = self._X
        self.y = self._y.ravel()
        self.featureName = self._featureNames

        # selected features and X after feature selection
        self.selectedFeatureNames = None
        self.selectedX = None

    def preprocess(self, y_is_binary=False, if_standardize=False, degree_of_polynomial=None,
                   balance_binary_outcome=False):
        """
        :param y_is_binary: (bool) set True if outcome is a binary variable
        :param if_standardize: (bool) set True to standardize features and outcome
        :param degree_of_polynomial: (int >=1 ) to add polynomial terms
        :param balance_binary_outcome: (bool) set True to balance the binary outcome
        """

        if if_standardize:
            self.X = standardize(self._X)
            if not y_is_binary:
                self.y = standardize(self._y.reshape(-1, 1)).ravel()

        if y_is_binary and balance_binary_outcome:
            sm = SMOTE(random_state=0)
            self.X, self.y = sm.fit_resample(self.X, self.y)

        if degree_of_polynomial is not None:
            poly = PolynomialFeatures(degree_of_polynomial)
            # polynomial is always done after standardization, so we work with X here
            self.X = poly.fit_transform(self.X)  # updating feature values
            self.featureName = poly.get_feature_names(self._featureNames)  # updating feature names

        # updating dataframe
        self.df = pd.DataFrame(self.X, columns=self.featureName)
        self.df[self._yName] = self.y

    def update_selected_features(self, selected_features):
        """ update the selected feature names and the selectedX
        :param selected_features: (list) of selected feature names
        """

        # update feature names
        self.selectedFeatureNames = selected_features
        # update predictor values
        self.selectedX = np.asarray(self.df[self.selectedFeatureNames])


def build_dataset(weeks_into_winter,
                  weeks_of_pred_period,
                  hosp_thresholds,
                  survey_size_novel_inf,
                  report_corr=True,
                  n_of_trajs_used=None):
    """ create the dataset needed to develop the predictive models
    :param weeks_into_winter: (int) a positive int for number of weeks into winter
    :param weeks_of_pred_period: (tuple) weeks when the prediction period starts and ends;
        it is assumed that prediction is made at time y0 for an outcome observed during (y0, y1).
    :param hosp_thresholds: (list) of thresholds of hospitalization capacity
    :param survey_size_novel_inf: (int) survey size of novel infection surveillance
    :param report_corr: (bool) whether to report correlations between features and outcomes
    :param n_of_trajs_used: (None or int) number of trajectories used to build the dataset
        (if None, all trajectories are used)
    """

    # read dataset
    feature_engineer = FeatureEngineering(
        dir_of_trajs='outputs/trajectories',
        weeks_of_pred_period=weeks_of_pred_period,
        hosp_thresholds=hosp_thresholds,
        n_of_trajs_used=n_of_trajs_used)

    # error model for novel variant surveillance
    err_novel_incd = ErrorModel(survey_size=survey_size_novel_inf, weeks_delay=1)

    # find output file name
    label = get_dataset_labels(week=weeks_into_winter, survey_size=survey_size_novel_inf)
    output_file = 'data-{}.csv'.format(label)

    # create new dataset based on raw data
    feature_engineer.pre_process(
        # information for incidence and prevalence features can be provided as
        #   'name of the feature' to calculate the recording at prediction time
        # or
        #   (feature's name, 100, ('ave', 2), ('slope', 4))
        # to multiply the column values as needed,
        # to report the recording at prediction time, and
        # to calculate the average and slope of observations during past weeks
        info_of_incd_fs=[
            # ('Obs: Incidence rate', 100000, ('ave', 2), ('slope', 4)),
            ('Obs: New hospitalization rate', 100000, ('ave', 2), ('slope', 4)),
            # ('Obs: % of new hospitalizations that are vaccinated', ('ave', 2), ('slope', 4)),
            ('Obs: % of incidence due to Novel', err_novel_incd, ('ave', 2), ('slope', 4)),
            # ('Obs: % of new hospitalizations due to novel variant', ('ave', 2), ('slope', 4)),
            # ('Obs: % of new hospitalizations due to Novel-V', ('ave', 2), ('slope', 4)),
        ],
        info_of_prev_fs=[
            ('Obs: Hospital occupancy rate', 100000),
            # ('Obs: Cumulative hospitalization rate', 100000),
            ('Obs: Cumulative vaccination rate', 1),
            # ('Obs: Prevalence susceptible', 1)
        ],
        info_of_parameter_fs=[
            # 'R0',
            # 'Duration of infectiousness-dominant',
            # 'Prob novel strain params-0',
            # 'Prob novel strain params-1',
            # 'Prob novel strain params-2',
            # 'Ratio infectiousness duration of novel to dominant',
            # 'Duration of vaccine immunity',
            # 'Ratio of duration of immunity from infection+vaccination to infection',
            # 'Vaccine effectiveness against infection-0',
            # 'Vaccine effectiveness against infection-1',
            # 'Ratio transmissibility of novel to dominant',
            # 'Vaccine effectiveness in reducing infectiousness-0',
            # 'Vaccine effectiveness in reducing infectiousness-1',
            # 'Ratio prob of hospitalization of novel to dominant',
            # 'Vaccine effectiveness against hospitalization-0',
            # 'Vaccine effectiveness against hospitalization-1',
            # 'Prob Hosp for 18-29',
            # 'Relative prob hosp by age-0',
            # 'Relative prob hosp by age-1',
            # 'Relative prob hosp by age-2',
            # 'Relative prob hosp by age-4',
            # 'Relative prob hosp by age-5',
            # 'Relative prob hosp by age-6',
            # 'Relative prob hosp by age-7',
            # 'Duration of E-0',
            # 'Duration of E-1',
            # 'Duration of E-2',
            # 'Duration of E-3',
            # 'Duration of I-0',
            # 'Duration of I-1',
            # 'Duration of I-2',
            # 'Duration of I-3',
            # 'Duration of Hosp-0',
            # 'Duration of Hosp-1',
            # 'Duration of Hosp-2',
            # 'Duration of Hosp-3',
            # 'Duration of R-0',
            # 'Duration of R-1',
            # 'Duration of R-2',
            # 'Duration of R-3',
            # 'Vaccination rate params-0',
            # 'Vaccination rate params-1',
            # 'Vaccination rate params-3',
            # 'Vaccination rate t_min by age-1',
            # 'Vaccination rate t_min by age-2',
            # 'Vaccination rate t_min by age-3',
            # 'Vaccination rate t_min by age-4',
            # 'Vaccination rate t_min by age-5',
            # 'Vaccination rate t_min by age-6',
            # 'Vaccination rate t_min by age-7',
            # 'Y1 thresholds-0',
            # 'Y1 thresholds-1',
            # 'Y1 Maximum hosp occupancy',
            # 'Y1 Max effectiveness of control measures'
        ],
        output_file=output_file,
        report_corr=report_corr)


def build_and_combine_datasets(
        name_of_dataset, first_week_of_winter, last_week_of_winter,
        weeks_to_predict, hosp_occu_thresholds, survey_size_novel_inf, n_of_trajs_used=None):
    """
    :param name_of_dataset: (string) name of the dataset to save as
    :param first_week_of_winter: (int) first week of winter
    :param last_week_of_winter: (int) last week of winter
    :param weeks_to_predict: (int) number of weeks in future when outcomes should be predicted
    :param hosp_occu_thresholds: (list) of thresholds for hospital occupancy
    :param survey_size_novel_inf: (int) survey size of novel infection surveillance
    :param n_of_trajs_used: (None or int) number of trajectories used to build the dataset
        (if None, all trajectories are used)
    """

    # datasets for predicting whether hospitalization capacities would surpass withing 4 weeks
    weeks_into_winter = 0
    weeks = []  # to then read the files
    while first_week_of_winter + weeks_into_winter + weeks_to_predict <= last_week_of_winter:

        weeks.append(weeks_into_winter)

        # the week at which the prediction should be made
        pred_week = first_week_of_winter + weeks_into_winter

        # build the dataset
        build_dataset(weeks_into_winter=weeks_into_winter,
                      weeks_of_pred_period=(pred_week, pred_week + weeks_to_predict),
                      hosp_thresholds=hosp_occu_thresholds,
                      survey_size_novel_inf=survey_size_novel_inf,
                      report_corr=False,
                      n_of_trajs_used=n_of_trajs_used)

        weeks_into_winter += 2

    # merge the data collected at different weeks to from a
    # single dataset for training the model
    dataframes = []
    prefix = '/outputs/prediction_datasets'
    for w in weeks:
        # file name
        label = get_dataset_labels(week=w, survey_size=survey_size_novel_inf)
        file_name = ROOT_DIR + prefix + '/data-{}.csv'.format(label)
        # read and store
        dataframes.append(pd.read_csv(file_name))
        # delete file
        delete_file(file_name=file_name)

    dataset = pd.concat(dataframes)
    dataset.to_csv(ROOT_DIR + prefix + '/{}.csv'.format(name_of_dataset),
                   index=False)

    # report the % of observations where hospital occupancy threshold passes
    print("---- summary of '{}' dataset ----".format(name_of_dataset))
    print('# of records: {}'.format(len(dataset)))
    for t in hosp_occu_thresholds:
        print('% observations where threshold {} is passed:'.format(t),
              round(100 * (1 - dataset[get_outcome_label(threshold=t)].mean()), 1))

    # report correlation
    outcomes = [get_outcome_label(threshold=t) for t in hosp_occu_thresholds]
    report_corrs(df=dataset,
                 outcomes=outcomes,
                 csv_file_name=ROOT_DIR + prefix + '/corr in {}.csv'.format(name_of_dataset))

