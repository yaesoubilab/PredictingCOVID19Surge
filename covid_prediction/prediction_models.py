import matplotlib.pyplot as plt
import numpy as np
import pydotplus
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import SimPy.Statistics as Stat


class Classifier:

    def __init__(self, df, features, y_name):

        self.features = features
        self.y_name = y_name
        self.df = df
        self.X = np.asarray(self.df[self.features])
        self.y = np.asarray(self.df[self.y_name])

        self.performanceTest = None  # performance summary on the test set

    def _update_binary_performance_plot_roc_curve(
            self, y_test, y_test_hat, y_test_hat_prob=None,
            model_name=None, display_roc_curve=True):
        """ update performance for classification models & plot roc curve """

        print('Needs to be debugged...')

        self.performanceTest = BinaryPerformanceSummary(y_test=y_test,
                                                        y_test_hat=y_test_hat,
                                                        y_test_hat_prob=y_test_hat_prob)
        if display_roc_curve:
            self.performanceTest.plot_roc_curve(model_name=model_name)


class TreePerformanceSummary:

    def __init__(self, y_test, y_test_hat):

        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sen = tp / (tp + fn)
        self.spe = tn / (tn + fp)
        self.accuracy = accuracy_score(y_test, y_test_hat)

    def print(self):
        print('Accuracy:', self.accuracy)
        print('Sensitivity:', self.sen)
        print('Specificity:', self.spe)


class DecisionTree(Classifier):

    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run(self, test_size=0.2, criterion="mse", max_depth=None, save_decision_path_filename=None):

        X = np.asarray(self.df[self.features])
        y = np.asarray(self.df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

        # fit model
        clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=1)
        clf.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = clf.predict(x_test)

        # update model performance attributes
        self.performanceTest = TreePerformanceSummary(y_test=y_test, y_test_hat=y_test_hat)

        if save_decision_path_filename is not None:
            self._plot_decision_path(model=clf, features=self.features, x_test=x_test,
                                     file_name=save_decision_path_filename)
            # # separate tree paths
            # i = 0
            # for datapoint in x_test:
            #     self.plot_decision_path(features=self.features, x_test=[datapoint],
            #                             file_name='{}tree{}.png'.format(output_path, i))
            #     i += 1

    @staticmethod
    def _plot_decision_path(model, features, x_test, file_name):
        # ref: https://stackoverflow.com/questions/55878247/how-to-display-the-path-of-a-decision-tree-for-test-samples
        # print graph of decision tree path
        # a visited node is colored in green, all other nodes are white.
        dot_data = export_graphviz(model,
                                   out_file=None, feature_names=features, class_names=['No', 'Yes'],
                                   filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)

        # # empty all nodes, i.e.set color to white and number of samples to zero
        # for node in graph.get_node_list():
        #     if node.get_attributes().get('label') is None:
        #         continue
        #     if 'samples = ' in node.get_attributes()['label']:
        #         labels = node.get_attributes()['label'].split('<br/>')
        #         for i, label in enumerate(labels):
        #             if label.startswith('samples = '):
        #                 labels[i] = 'samples = 0'
        #         node.set('label', '<br/>'.join(labels))
        #         node.set_fillcolor('white')
        #
        # samples = x_test
        # decision_paths = self.model.decision_path(samples)
        #
        # for decision_path in decision_paths:
        #     for n, node_value in enumerate(decision_path.toarray()[0]):
        #         if node_value == 0:
        #             continue
        #         node = graph.get_node(str(n))[0]
        #         node.set_fillcolor('green')
        #         labels = node.get_attributes()['label'].split('<br/>')
        #         for i, label in enumerate(labels):
        #             if label.startswith('samples = '):
        #                 labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)
        #
        #         node.set('label', '<br/>'.join(labels))

        graph.write_png(file_name)



# --------------------------------------------


class MultiClassifiers:

    def __init__(self, df, features, y_name):

        self.df = df
        self.features = features
        self.y_name = y_name
        self.performancesTest = None

    def _update_binary_performances_plot_roc_curve(self, performance_list, display_roc_curve):
        self.performancesTest = BootstrapBinaryPerformanceSummary(performance_list=performance_list)
        if display_roc_curve:
            self.performancesTest.bootstrap_roc_curve()

    def _update_decision_tree_performances(self, performance_list):
        self.performancesTest = BootstrapTreePerformanceSummary(performance_list=performance_list)


class MultiDecisionTrees(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, test_size=0.2, penalty='l2', l1_solver='liblinear', display_roc_curve=True):

        performance_test_list = []
        i = 0
        model = DecisionTree(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(test_size=test_size, save_decision_path_filename=False)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self._update_decision_tree_performances(performance_list=performance_test_list)


# ----------------------------------------

class LinearReg(Classifier):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

        self.selected_features = None

    def run(self, random_state, test_size=0.2,
            penalty='none', alpha=0.1, cv=False):
        """
        run linear regression model
        :param test_size: size of the test set
        :param random_state: random state number
        :param penalty: 'none', 'l1', or 'l2'
        :param alpha: the degree of sparsity of the estimated coefficients for Lasso or Ridge
        :param cv: whether we use cross-validation and visualize prediction errors
        """

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        # specify model
        if penalty == 'l1':
            reg = linear_model.Lasso(alpha=alpha)
        elif penalty == 'l2':
            reg = linear_model.Ridge(alpha=alpha)
        else:
            reg = linear_model.LinearRegression()

        # fit model
        reg.fit(X=x_train, y=y_train)
        # prediction
        y_test_hat = reg.predict(x_test)

        # cross validation
        cv_score = cross_val_score(estimator=reg, X=self.X, y=self.y)

        # update performance
        self.performanceTest = LinearPerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, cv_score=cv_score)

        # if cv:
        #     plot_cv_graph(reg=reg, x=self.X, y=self.y)


class MultiLinearReg(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, test_size=0.2, penalty='none', alpha=0.1):

        performance_test_list = []
        i = 0
        model = LinearReg(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(random_state=i, test_size=test_size, penalty=penalty, alpha=alpha)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self.performancesTest = BootstrapLinearPerformanceSummary(performance_list=performance_test_list)


class LogisticReg(Classifier):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

        self.coeffs = None
        self.intercept = None

    def run(self, random_state,
            test_size=0.2, penalty='l2', l1_solver='liblinear', C=0.1, display_roc_curve=True):
        """
        :param random_state: random state number
        :param l1_solver: solver that handle 'l1' penalty. 'liblinear' good for small dataset, 'sage' good for large set
        :param C: inverse of regularization strength, must be positive
        :param test_size: size of test sample
        :param penalty: 'l1','l2', or 'none' (default 'l2')
        :param display_roc_curve: whether plot roc curve
        """

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        # fit model
        solver = 'lbfgs'
        if penalty == 'l1':
            solver = l1_solver
        LR = linear_model.LogisticRegression(penalty=penalty, solver=solver, C=C)
        LR.fit(X=x_train, y=y_train)

        # coefficients
        self.coeffs = LR.coef_
        self.intercept = LR.intercept_

        # prediction
        y_test_hat = LR.predict(x_test)
        y_test_hat_prob = LR.predict_proba(x_test)

        # update model performance attributes
        self._update_binary_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob,
            model_name='logistic regression', display_roc_curve=display_roc_curve)


class MultiLogisticReg(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps,
                 test_size=0.2, penalty='l2', l1_solver='liblinear', C=0.1, display_roc_curve=True):
        """
        :param penalty: 'l1','l2', or 'none' (default 'l2')
        """

        performance_test_list = []
        i = 0
        model = LogisticReg(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(random_state=i, test_size=test_size,
                      penalty=penalty, l1_solver=l1_solver, C=C, display_roc_curve=False)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self._update_binary_performances_plot_roc_curve(performance_list=performance_test_list,
                                                        display_roc_curve=display_roc_curve)


class NNRegression(Classifier):
    def __init__(self, df, features, y_name, len_neurons=None):
        super().__init__(df, features, y_name)

        self.len_neurons = len_neurons if len_neurons is not None else len(self.features) + 2

    def run(self, random_state,
            activation='logistic', solver='sgd', alpha=0.0001, max_iter=1000, test_size=0.2):
        if test_size == 0:
            x_train = self.X
            x_test = self.X
            y_train = self.y
            y_test = self.y
        else:
            # split train vs. test set
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                test_size=test_size,
                                                                random_state=random_state)

        # fit model
        clf = MLPRegressor(alpha=alpha,  # alpha: l2 penalty (regularization)
                           max_iter=max_iter,
                           hidden_layer_sizes=(self.len_neurons, ),
                           random_state=random_state,
                           solver=solver,  # the default 'adam' is preferred for large dataset
                           activation=activation)
        clf.fit(X=x_train, y=y_train)

        # plt.plot(clf.loss_curve_)  # plotting by columns
        # plt.show()

        # prediction
        y_test_hat = clf.predict(X=x_test)

        # cross validation
        cv_score = cross_val_score(estimator=clf, X=self.X, y=self.y)

        # update model performance attributes
        self.performanceTest = LinearPerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, cv_score=cv_score)


class MultiNNRegression(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, activation='logistic', solver='sgd',
                 alpha=0.0001, max_iter=1000, test_size=0.2):
        performance_test_list = []
        i = 0
        model = NNRegression(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(random_state=i, test_size=test_size,
                      activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self.performancesTest = BootstrapLinearPerformanceSummary(performance_list=performance_test_list)


class NNClassification(Classifier):
    def __init__(self, df, features, y_name, len_neurons=None):
        super().__init__(df, features, y_name)

        self.len_neurons = len_neurons

    def run(self, random_state, test_size=0.2, activation='logistic', solver='adam', display_roc_curve=True):
        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        # fit model
        if self.len_neurons is None:
            self.len_neurons = len(self.features) + 2
        clf = MLPClassifier(alpha=0.000001,         # alpha: l2 penalty (regularization)
                            max_iter=10000,
                            hidden_layer_sizes=(self.len_neurons,),
                            random_state=random_state,
                            solver=solver,     # the default 'adam' is preferred for large dataset
                            activation=activation)
        clf.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = clf.predict(X=x_test)
        y_test_hat_prob = clf.predict_proba(X=x_test)

        # update model performance attributes
        self._update_binary_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob,
            model_name='neural network', display_roc_curve=display_roc_curve)


class PerformanceSummary:
    def __init__(self, y_test, y_test_hat, y_test_hat_prob=None):
        """
        :param y_test: list of true ys for model validation
        :param y_test_hat: list of predicted ys (binary)
        """
        self.y_test = y_test
        self.y_test_hat = y_test_hat
        self.y_test_hat_prob = y_test_hat_prob


class BootstrapPerformanceSummary:
    def __init__(self, performance_list):
        """ performance summary for multiple models """
        self.performances = performance_list


class LinearPerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, cv_score):
        super().__init__(y_test, y_test_hat)
        self.r2 = r2_score(y_true=y_test, y_pred=y_test_hat)
        self.mse = mean_squared_error(y_true=y_test, y_pred=y_test_hat)
        self.cv = cv_score
        # print(self.cv.mean())
        # self.coefficient = model.coef_
        # self.intercept = model.intercept_

    def print(self):
        # print('Coefficients:', self.coefficient)
        # print('Intercept:', self.intercept)
        print('R2:', self.r2)
        print('MSE:', self.mse)
        print('cross-validation score:', self.cv.mean())


class BootstrapLinearPerformanceSummary(BootstrapPerformanceSummary):
    def __init__(self, performance_list):
        super().__init__(performance_list=performance_list)
        self.statR2 = Stat.SummaryStat(name="R-square", data=[performance.r2 for performance in self.performances])
        self.statMSE = Stat.SummaryStat(name='MSE', data=[performance.mse for performance in self.performances])
        self.statCV = Stat.SummaryStat(name='cross-validation',
                                       data=[performance.cv.mean() for performance in self.performances])

    def print(self, decimal=3):
        print('R2:', self.statR2.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('CV-R2:', self.statCV.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        # print('MSE:', self.statMSE.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))


class BinaryPerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, y_test_hat_prob):
        super().__init__(y_test, y_test_hat, y_test_hat_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sen = tp / (tp + fn)
        self.spe = tn / (tn + fp)
        self.fpr, self.tpr, threshold = roc_curve(y_test, y_test_hat_prob[:, 1], drop_intermediate=False)
        self.roc_auc = auc(self.fpr, self.tpr)

    def print(self):
        # print("Sensitivity:", self.sen)
        # print("Specificity:", self.spe)
        print("AUC:", self.roc_auc)

    def plot_roc_curve(self, model_name):
        fpr, tpr, threshold = roc_curve(self.y_test, self.y_test_hat_prob[:, 1])
        plt.plot(fpr, tpr, color='green', lw=1, alpha=1)
        plt.plot([0, 1], [0, 1], color='blue', lw=1, alpha=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for {} model'.format(model_name))
        plt.text(0.7, 0.1, 'AUC: {}'.format(round(self.roc_auc, 2)))
        plt.show()


class BootstrapBinaryPerformanceSummary(BootstrapPerformanceSummary):
    def __init__(self, performance_list):
        super().__init__(performance_list=performance_list)
        self.statSen = Stat.SummaryStat(name="sensitivity", data=[performance.sen for performance in self.performances])
        self.statSpe = Stat.SummaryStat(name="sensitivity", data=[performance.spe for performance in self.performances])
        self.statAUC = Stat.SummaryStat(name='roc-auc', data=[performance.roc_auc for performance in self.performances])

    def print(self, decimal=3):
        # print('Sensitivity:', self.statSen.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        # print('Specificity:', self.statSpe.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('ROC-AUC:', self.statAUC.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))

    def bootstrap_roc_curve(self):
        auc_mean = self.statAUC.get_mean()
        auc_PI = self.statAUC.get_PI(alpha=0.05)
        plt.figure(figsize=(5, 5))
        for performance in self.performances:
            fpr, tpr, threshold = roc_curve(performance.y_test, performance.y_test_hat_prob[:, 1])
            plt.plot(fpr, tpr, color='lightskyblue', lw=1, alpha=0.5)
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('AUC-ROC for logistic regression models')
        plt.text(0.95, 0.05,
                 "Area:{} ({}, {})".format(round(auc_mean, 2),
                                           round(auc_PI[0], 2), round(auc_PI[1], 2)),
                 ha="right", va="bottom", fontsize=10)
        plt.show()





class BootstrapTreePerformanceSummary(BootstrapPerformanceSummary):
    def __init__(self, performance_list):
        super().__init__(performance_list=performance_list)
        self.statSen = Stat.SummaryStat(name="sensitivity", data=[performance.sen for performance in self.performances])
        self.statSpe = Stat.SummaryStat(name="sensitivity", data=[performance.spe for performance in self.performances])

    def print(self, decimal=3):
        print('Sensitivity:', self.statSen.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('Specificity:', self.statSpe.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))


def plot_cv_graph(reg, x, y):
    predicted = cross_val_predict(reg, x, y, cv=10)
    # visualization
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()