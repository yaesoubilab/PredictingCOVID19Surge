import multiprocessing as mp

from SimPy.InOutFunctions import write_csv
from SimPy.Statistics import SummaryStat
from covid_prediction.pre_process import *
from covid_prediction.prediction_models import *

MAX_PROCESSES = mp.cpu_count()  # maximum number of processors


class _CrossValidSummary:
    """ cross validation and specification information """
    def __init__(self, n_features):

        self.nFeatures = n_features

        self.scores = None
        self.summaryStat = None
        self.selectedFeatures = None
        self.meanScore = None
        self.error = None
        self.PI = None
        self.formattedMeanPI = None

    def add_cv_performance(self, scores, deci=4, selected_features=None):
        """
        gets the list of scores to calculate summary statistics (mean, percentile intervals, and error length)
        :param scores: (list) of scores from cross validation
        :param deci: (int) number of digits to round the summary statistics to
        :param selected_features: (list) of selected features
        """
        self.scores = scores
        self.summaryStat = SummaryStat(name='cross-validation scores',
                                       data=scores)
        self.selectedFeatures = selected_features
        self.meanScore = self.summaryStat.get_mean()
        self.PI = self.summaryStat.get_PI(alpha=0.05)
        self.error = 0.5*(self.PI[1] - self.PI[0])
        self.formattedMeanPI = self.summaryStat.get_formatted_mean_and_interval(deci=deci, interval_type='c')


class LinRegCVSummary(_CrossValidSummary):
    """ linear regression cross-validation performance summary """

    def __init__(self, penalty, poly_degree, n_features, f_s_method):

        super().__init__(n_features=n_features)
        self.penalty = penalty
        self.poly_degree = poly_degree
        self.n_fs = n_features
        self.f_s_method = f_s_method    # feature selection method


class NeuralNetCVSummary(_CrossValidSummary):
    """ neural network cross-validation performance summary """

    def __init__(self, n_features, alpha, n_neurons):

        super().__init__(n_features=n_features)
        self.alpha = alpha
        self.nNeurons = n_neurons


class NeuralNetCrossValidator:
    """ class to run cross validation on a neural network model """

    def __init__(self, preprocessed_data,
                 n_features_wanted, alpha, n_neurons,
                 feature_selection_method, cv_fold):
        """
        :param preprocessed_data: (PreProcessor)
        :param n_features_wanted:
        :param alpha:
        :param n_neurons:
        :param feature_selection_method:
        :param cv_fold:
        """
        assert isinstance(preprocessed_data, PreProcessor)

        self.preProcessedData = preprocessed_data
        self.nFeatures = n_features_wanted
        self.alpha = alpha
        self.nNeurons = n_neurons
        self.featureSelection = feature_selection_method
        self.cvFold = cv_fold

        self.performanceSummary = None

    def go(self):
        """ performs cross validation and calculates the R2 scores """

        # make a performance object
        self.performanceSummary = NeuralNetCVSummary(
            n_features=self.nFeatures, alpha=self.alpha, n_neurons=self.nNeurons)

        # construct a neural network model
        model = MLPRegressor(alpha=self.alpha, hidden_layer_sizes=(self.nNeurons,),
                             max_iter=1000, solver='sgd', activation='logistic',
                             random_state=0)

        # feature selection
        self.preProcessedData.feature_selection(
            estimator=model, method=self.featureSelection, num_fs_wanted=self.nFeatures)

        # cross-validation
        cv_score_list = cross_val_score(estimator=model,
                                        X=self.preProcessedData.selectedX,
                                        y=self.preProcessedData.y,
                                        cv=self.cvFold)

        # store the performance of this specification
        self.performanceSummary.add_cv_performance(scores=cv_score_list,
                                                   deci=4,
                                                   selected_features=self.preProcessedData.selectedFeatureNames)


def run_this_cross_validator(cross_validator, i):
    """ helper function for parallelization  """

    # simulate and return the cohort
    cross_validator.go()
    return cross_validator


class NeuralNetSepecOptimizer:
    """ class to find the optimal specification for a neural network using cross validation """

    def __init__(self, data, feature_names, outcome_name,
                 list_of_n_features_wanted, list_of_alphas, list_of_n_neurons,
                 feature_selection_method, cv_fold, if_standardize=True):

        self.crossValidators = []
        self.crossValidationSummaries = []

        # preprocess
        preprocessed_data = PreProcessor(df=data, feature_names=feature_names, y_name=outcome_name)
        preprocessed_data.preprocess(if_standardize=if_standardize)

        # send the number of neurons to (number of features + 2)
        # if the number of neurons is not provided
        if list_of_n_neurons is None:
            list_of_n_neurons = [len(feature_names) + 2]

        for n_fs in list_of_n_features_wanted:
            for alpha in list_of_alphas:
                for n_neurons in list_of_n_neurons:
                    self.crossValidators.append(
                        NeuralNetCrossValidator(
                            preprocessed_data=preprocessed_data,
                            n_features_wanted=n_fs, alpha=alpha, n_neurons=n_neurons,
                            feature_selection_method=feature_selection_method, cv_fold=cv_fold))

    def find_best_spec(self, run_in_parallel=False, save_to_file_performance=None, save_to_file_features=None):
        """ find the best specification for the neural network model
        :param run_in_parallel: (bool) set to True to run the cross validation in parallel
        :param save_to_file_performance: (string) filename where the performance results should be saved.
        :param save_to_file_features: (string) filename where the selected features should be saved
        :return: the best specification
        """

        if run_in_parallel:

            # create a list of arguments
            args = [(cv, 0) for cv in self.crossValidators]

            # run all
            with mp.Pool(MAX_PROCESSES) as pl:
                analyzed_cross_validators = pl.starmap(run_this_cross_validator, args)

            for cv in analyzed_cross_validators:
                self.crossValidationSummaries.append(cv.performanceSummary)

        else:
            for cv in self.crossValidators:
                cv.go()
                self.crossValidationSummaries.append(cv.performanceSummary)

        # find the best specification
        best_spec = None
        max_r2 = float('-inf')
        summary = [['# features', 'alpha', '# neurons', 'R2', 'R2 and PI']]
        for s in self.crossValidationSummaries:
            summary.append([s.nFeatures, s.alpha, s.nNeurons, s.meanScore, s.formattedMeanPI])
            if s.meanScore > max_r2:
                best_spec = s
                max_r2 = s.meanScore

        # print score of each specification
        if save_to_file_performance is not None:
            write_csv(rows=summary, file_name=save_to_file_performance)

        if save_to_file_features is not None:
            write_csv(rows=best_spec.selectedFeatures, file_name=save_to_file_features)

        return best_spec


