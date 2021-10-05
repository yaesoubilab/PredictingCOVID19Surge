import multiprocessing as mp

from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeClassifier

from SimPy.InOutFunctions import write_csv
from SimPy.Statistics import SummaryStat
from covid_prediction.pre_process import PreProcessor
from covid_prediction.prediction_models import DecisionTree

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

    def __init__(self, n_features, max_depth, ccp_alpha):

        super().__init__(n_features=n_features)
        self.maxDepth = max_depth
        self.ccpAlpha = ccp_alpha


class _CrossValidator:
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
        cv_scores = cross_val_score(estimator=model,
                                    X=self.preProcessedData.selectedX,
                                    y=self.preProcessedData.y,
                                    cv=self.cvFold,
                                    scoring=self.scoring)

        # store the performance of this specification
        self.performanceSummary.add_cv_performance(scores=cv_scores,
                                                   deci=2,
                                                   selected_features=self.preProcessedData.selectedFeatureNames)


class NeuralNetCrossValidator(_CrossValidator):
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

        _CrossValidator.__init__(self,
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


class DecTreeCrossValidator(_CrossValidator):
    """ class to run cross validation on a decision tree model """

    def __init__(self, preprocessed_data, cv_fold, scoring,
                 feature_selection_method=None, n_features_wanted=None,
                 max_depth=None, ccp_alpha=0.0):
        """
        :param preprocessed_data: (PreProcessor)
        :param n_features_wanted: (int or None)
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :param feature_selection_method: (string or None) 'rfe', 'lasso', or 'pi'
        :param max_depth: (int) maximum depth of the decision tree
        :param ccp_alpha: (float) Complexity parameter used for Minimal Cost-Complexity Pruning
        """

        _CrossValidator.__init__(self,
                                 preprocessed_data=preprocessed_data,
                                 n_features_wanted=n_features_wanted,
                                 feature_selection_method=feature_selection_method,
                                 cv_fold=cv_fold, scoring=scoring)

        self.maxDepth = max_depth
        self.ccpAlpha = ccp_alpha

    def go(self):
        """ performs cross validation and calculates the scores """

        # make a performance object
        self.performanceSummary = DecTreeCVSummary(
            n_features=self.nFeatures, max_depth=self.maxDepth, ccp_alpha=self.ccpAlpha)

        # construct a decision tree model
        model = DecisionTreeClassifier(max_depth=self.maxDepth, ccp_alpha=self.ccpAlpha, random_state=0)

        # fit the model to find the strongest features on the entire dataset
        model.fit(X=self.preProcessedData.X, y=self.preProcessedData.y)

        # get selected features
        selected_features = []
        for i, v in enumerate(model.feature_importances_):
            if v > 0:
                selected_features.append(self.preProcessedData.featureName[i])

        # update selected features and predictors
        self.preProcessedData.update_selected_features(selected_features=selected_features)

        # perform cross validation on this model
        cv_scores = cross_val_score(estimator=model,
                                    X=self.preProcessedData.selectedX,
                                    y=self.preProcessedData.y,
                                    cv=self.cvFold,
                                    scoring=self.scoring)

        # store the performance of this specification
        self.performanceSummary.add_cv_performance(scores=cv_scores,
                                                   deci=2,
                                                   selected_features=self.preProcessedData.selectedFeatureNames)


def run_this_cross_validator(cross_validator, i):
    """ helper function for parallelization (the extra argument i is needed to prevent errors) """

    # simulate and return the cohort
    cross_validator.go()
    return cross_validator


class _ParameterOptimizer:
    """ class to find the optimal parameters for a model using cross validation """

    def __init__(self, df, feature_names, outcome_name, if_outcome_binary, if_standardize=True):
        """
        :param df: (panda DataFrame)
        :param feature_names: (list) of feature names to be included in the analysis
        :param outcome_name: (string) name of the outcome
        :param if_outcome_binary: (bool) if outcome is binary
        :param if_standardize: (bool) if inputs should be standardized
        """

        self.crossValidators = []
        self.crossValidationSummaries = []

        # preprocess
        self.preprocessedData = PreProcessor(df=df, feature_names=feature_names, y_name=outcome_name)
        self.preprocessedData.preprocess(y_is_binary=if_outcome_binary, if_standardize=if_standardize)

    def _run(self, run_in_parallel):
        """ runs cross validation over all combinations of parameters """

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

    def _save_results(self, summary, best_spec, save_to_file_performance, save_to_file_features):
        """
        prints the results to csv files
        :param summary: (list) of performance summary to be saved
        :param best_spec: (_CrossValidSummary) the best specifications for the model
        :param save_to_file_performance: (string) filename to save the performance summary as
        :param save_to_file_features: (string) filename to save the selected features as
        """

        # save the score of each specification
        if save_to_file_performance is not None:
            write_csv(rows=summary, file_name=save_to_file_performance)

        if save_to_file_features is not None:
            write_csv(rows=[[f] for f in best_spec.selectedFeatures],
                      file_name=save_to_file_features)


class NeuralNetParameterOptimizer(_ParameterOptimizer):
    """ class to find the optimal parameters for a neural network using cross validation """

    def __init__(self, df, feature_names, outcome_name, if_outcome_binary,
                 list_of_n_features_wanted, list_of_alphas, list_of_n_neurons,
                 feature_selection_method, cv_fold, scoring=None, if_standardize=True):
        """
        :param df: (panda DataFrame)
        :param feature_names: (list) of feature names to be included in the analysis
        :param outcome_name: (string) name of the outcome
        :param if_outcome_binary: (bool) if outcome is binary
        :param list_of_n_features_wanted: (list) of number of features wanted
        :param list_of_alphas: (list) of alphas
        :param list_of_n_neurons: (list) of number of neurons
        :param feature_selection_method: (string) 'rfe', 'lasso', or 'pi'
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        :param if_standardize: (bool) if inputs should be standardized
        """

        _ParameterOptimizer.__init__(self, df=df,
                                     feature_names=feature_names,
                                     outcome_name=outcome_name,
                                     if_outcome_binary=if_outcome_binary,
                                     if_standardize=if_standardize)

        # set the number of neurons to (number of features + 2)
        # if the number of neurons is not provided
        if list_of_n_neurons is None:
            list_of_n_neurons = [len(feature_names) + 2]

        for n_fs in list_of_n_features_wanted:
            for alpha in list_of_alphas:
                for n_neurons in list_of_n_neurons:
                    self.crossValidators.append(
                        NeuralNetCrossValidator(
                            preprocessed_data=self.preprocessedData,
                            n_features_wanted=n_fs, alpha=alpha, n_neurons=n_neurons,
                            feature_selection_method=feature_selection_method,
                            cv_fold=cv_fold, scoring=scoring))

    def find_best_parameters(self, run_in_parallel=False, save_to_file_performance=None, save_to_file_features=None):
        """ find the best specification for the neural network model
        :param run_in_parallel: (bool) set to True to run the cross validation in parallel
        :param save_to_file_performance: (string) filename where the performance results should be saved.
        :param save_to_file_features: (string) filename where the selected features should be saved
        :return: the best specification
        """

        # run
        self._run(run_in_parallel=run_in_parallel)

        # find the best specification
        best_spec = None
        max_r2 = float('-inf')
        summary = [['# features', 'alpha', '# neurons', 'Score', 'Score and PI']]
        for s in self.crossValidationSummaries:
            summary.append([s.nFeatures, s.alpha, s.nNeurons, s.meanScore, s.formattedMeanPI])
            if s.meanScore > max_r2:
                best_spec = s
                max_r2 = s.meanScore

        self._save_results(summary=summary, best_spec=best_spec,
                           save_to_file_performance=save_to_file_performance,
                           save_to_file_features=save_to_file_features)

        return best_spec


class DecTreeParameterOptimizer(_ParameterOptimizer):
    """ class to find the optimal parameters for a decision tree using cross validation """

    def __init__(self, df, feature_names, outcome_name, if_outcome_binary,
                 list_of_n_features_wanted, list_of_max_depths, list_of_ccp_alphas,
                 feature_selection_method, cv_fold, scoring=None):
        """
        :param df: (panda DataFrame)
        :param feature_names: (list) of feature names to be included in the analysis
        :param outcome_name: (string) name of the outcome
        :param if_outcome_binary: (bool) if outcome is binary
        :param list_of_n_features_wanted: (list) of number of features wanted
        :param list_of_max_depths: (list) of maximum depths
        :param list_of_ccp_alphas: (list) of ccp alphas
        :param feature_selection_method: (string) 'rfe', 'lasso', or 'pi'
        :param cv_fold: (int) number of cross validation folds
        :param scoring: (string) from: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        """

        _ParameterOptimizer.__init__(self, df=df,
                                     feature_names=feature_names,
                                     outcome_name=outcome_name,
                                     if_outcome_binary=if_outcome_binary,
                                     if_standardize=False)

        for n_fs in list_of_n_features_wanted:
            for max_depth in list_of_max_depths:
                for alpha in list_of_ccp_alphas:
                    self.crossValidators.append(
                        DecTreeCrossValidator(
                            preprocessed_data=self.preprocessedData,
                            feature_selection_method=feature_selection_method,
                            n_features_wanted=n_fs, cv_fold=cv_fold,
                            scoring=scoring, max_depth=max_depth, ccp_alpha=alpha))

    def find_best_parameters(self, run_in_parallel=False, save_to_file_performance=None, save_to_file_features=None):
        """ find the best specification for the neural network model
        :param run_in_parallel: (bool) set to True to run the cross validation in parallel
        :param save_to_file_performance: (string) filename where the performance results should be saved.
        :param save_to_file_features: (string) filename where the selected features should be saved
        :return: the best specification
        """

        # run
        self._run(run_in_parallel=run_in_parallel)

        # find the best specification
        best_spec = None
        max_score = float('-inf')
        summary = [['# features', 'max depth', 'ccp alpha', 'Score', 'Score and PI']]
        for s in self.crossValidationSummaries:
            summary.append([s.nFeatures, s.maxDepth, s.ccpAlpha, s.meanScore, s.formattedMeanPI])
            if s.meanScore > max_score:
                best_spec = s
                max_score = s.meanScore

        self._save_results(summary=summary, best_spec=best_spec,
                           save_to_file_performance=save_to_file_performance,
                           save_to_file_features=save_to_file_features)

        return best_spec

    @staticmethod
    def evaluate_tree_on_validation_set(df_training, df_validation, selected_features, y_name, max_depth, ccp_alpha):
        """
        :param df_training:
        :param df_validation:
        :param selected_features:
        :param y_name:
        :param max_depth:
        :param ccp_alpha:
        :return: the trained model (with performance calculated on the validation dataset)
        """

        # make a decision tree model
        model = DecisionTree(df=df_training, feature_names=selected_features, y_name=y_name)

        # train the model
        model.run(max_depth=max_depth, ccp_alpha=ccp_alpha, df_validation=df_validation)

        # return the trained model
        return model
