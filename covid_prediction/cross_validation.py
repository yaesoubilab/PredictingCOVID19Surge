import multiprocessing as mp

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from SimPy.InOutFunctions import write_csv
from SimPy.Statistics import SummaryStat
from covid_prediction.pre_process import PreProcessor

MAX_PROCESSES = mp.cpu_count()  # maximum number of processors


class _CrossValidSummary:
    """ cross validation results """
    def __init__(self, n_features):
        """
        :param n_features: (int) number of features
        """

        self.nFeatures = n_features

        self.scores = None  # list of scores (R2, ROC_AUC, etc.) from cross validation iterations
        self.summaryStat = None  # summary statistics of scores
        self.selectedFeatures = None  # features uses in the model evaluated in cross validation
        self.meanScore = None   # mean of scores
        self.PI = None
        self.formattedMeanPI = None  # formatted mean and percentile interval for the score

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
        self.formattedMeanPI = self.summaryStat.get_formatted_mean_and_interval(deci=deci, interval_type='c')


class LinRegCVSummary(_CrossValidSummary):
    """ results of linear regression cross-validation """

    def __init__(self, penalty, poly_degree, n_features, f_s_method):

        super().__init__(n_features=n_features)
        self.penalty = penalty
        self.poly_degree = poly_degree
        self.n_fs = n_features
        self.f_s_method = f_s_method    # feature selection method


class NeuralNetCVSummary(_CrossValidSummary):
    """ results of neural network cross-validation """

    def __init__(self, n_features, alpha, n_neurons):

        super().__init__(n_features=n_features)
        self.alpha = alpha
        self.nNeurons = n_neurons


class DecTreeCVSummary(_CrossValidSummary):
    """ results of decision tree cross-validation """

    def __init__(self, n_features, max_depth):
        # TODO: max_depth is a parameter that could be optimized through cross-validation,
        #   what are the other parameters of decision trees that should be optimize?

        super().__init__(n_features=n_features)
        self.maxDepth = max_depth


class CrossValidator:
    """ class to run cross validation """

    def __init__(self, preprocessed_data, n_features_wanted, feature_selection_method, cv_fold, scoring):
        """
        :param preprocessed_data: (PreProcessor)
        :param n_features_wanted: (int)
        :param feature_selection_method: (string) 'rfe', 'lasso', or 'pi'
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """

        assert isinstance(preprocessed_data, PreProcessor)

        self.preProcessedData = preprocessed_data
        self.nFeatures = n_features_wanted
        self.featureSelection = feature_selection_method
        self.cvFold = cv_fold
        self.scoring = scoring
        self.performanceSummary = None

    def _do_cross_validation(self, model):
        """
        performs cross validation on the provided model
        :param model: (regression or classification model)
        """

        # feature selection
        self.preProcessedData.feature_selection(
            estimator=model, method=self.featureSelection, num_fs_wanted=self.nFeatures)

        # cross-validation
        cv_score_list = cross_val_score(estimator=model,
                                        X=self.preProcessedData.selectedX,
                                        y=self.preProcessedData.y,
                                        cv=self.cvFold,
                                        scoring=self.scoring)

        # store the performance of this specification
        self.performanceSummary.add_cv_performance(scores=cv_score_list,
                                                   deci=2,
                                                   selected_features=self.preProcessedData.selectedFeatureNames)


class NeuralNetCrossValidator(CrossValidator):
    """ class to run cross validation on a neural network model """

    def __init__(self, preprocessed_data, feature_selection_method, cv_fold, scoring, n_features_wanted,
                 alpha, n_neurons):
        """
        :param preprocessed_data: (PreProcessor)
        :param n_features_wanted: (int)
        :param feature_selection_method: (string) 'rfe', 'lasso', or 'pi'
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :param alpha:
        :param n_neurons:
        """

        CrossValidator.__init__(self,
                                preprocessed_data=preprocessed_data,
                                n_features_wanted=n_features_wanted,
                                feature_selection_method=feature_selection_method,
                                cv_fold=cv_fold, scoring=scoring)

        self.alpha = alpha
        self.nNeurons = n_neurons

    def go(self):
        """ performs cross validation and calculates the scores """

        # make a performance object
        self.performanceSummary = NeuralNetCVSummary(
            n_features=self.nFeatures, alpha=self.alpha, n_neurons=self.nNeurons)

        # construct a neural network model
        model = MLPRegressor(alpha=self.alpha, hidden_layer_sizes=(self.nNeurons,),
                             max_iter=1000, solver='sgd', activation='logistic',
                             random_state=0)

        # perform cross validation on this model
        self._do_cross_validation(model=model)


class DecTreeCrossValidator(CrossValidator):
    """ class to run cross validation on a decision tree model """

    def __init__(self, preprocessed_data, feature_selection_method, cv_fold, scoring, n_features_wanted,
                 max_depth):
        """
        :param preprocessed_data: (PreProcessor)
        :param n_features_wanted: (int)
        :param feature_selection_method: (string) 'rfe', 'lasso', or 'pi'
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :param max_depth: (int) maximum depth of the decision tree
        """

        CrossValidator.__init__(self,
                                preprocessed_data=preprocessed_data,
                                n_features_wanted=n_features_wanted,
                                feature_selection_method=feature_selection_method,
                                cv_fold=cv_fold, scoring=scoring)

        self.maxDepth = max_depth

    def go(self):
        """ performs cross validation and calculates the scores """

        # make a performance object
        self.performanceSummary = DecTreeCVSummary(
            n_features=self.nFeatures, max_depth=self.maxDepth)

        # construct a decision tree model
        model = DecisionTreeClassifier(max_depth=self.maxDepth, random_state=0)

        # perform cross validation on this model
        self._do_cross_validation(model=model)


def run_this_cross_validator(cross_validator, i):
    """ helper function for parallelization (the extract argument i is to prevent errors) """

    # simulate and return the cohort
    cross_validator.go()
    return cross_validator


class NeuralNetSpecOptimizer:
    """ class to find the optimal specification for a neural network using cross validation """

    def __init__(self, data, feature_names, outcome_name, if_outcome_binary,
                 list_of_n_features_wanted, list_of_alphas, list_of_n_neurons,
                 feature_selection_method, cv_fold, scoring=None, if_standardize=True):

        self.crossValidators = []
        self.crossValidationSummaries = []

        # preprocess
        preprocessed_data = PreProcessor(df=data, feature_names=feature_names, y_name=outcome_name)
        preprocessed_data.preprocess(y_is_binary=if_outcome_binary, if_standardize=if_standardize)

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
                            feature_selection_method=feature_selection_method,
                            cv_fold=cv_fold, scoring=scoring))

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
        summary = [['# features', 'alpha', '# neurons', 'Score', 'Score and PI']]
        for s in self.crossValidationSummaries:
            summary.append([s.nFeatures, s.alpha, s.nNeurons, s.meanScore, s.formattedMeanPI])
            if s.meanScore > max_r2:
                best_spec = s
                max_r2 = s.meanScore

        # print score of each specification
        if save_to_file_performance is not None:
            write_csv(rows=summary, file_name=save_to_file_performance)

        if save_to_file_features is not None:
            write_csv(rows=[[f] for f in best_spec.selectedFeatures],
                      file_name=save_to_file_features)

        return best_spec


