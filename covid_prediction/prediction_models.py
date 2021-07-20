import matplotlib.pyplot as plt
import numpy as np
import pydotplus
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import SimPy.Statistics as Stat


def standardize(x):
    return StandardScaler().fit_transform(x)


class Classifier:
    def __init__(self, df, features, y_name):
        self.features = features
        self.y_name = y_name
        self.df = df
        self.performanceTest = None

    def _update_linear_performance(self, y_test, y_test_hat, model):
        """ update model performance for linear regression model & print coefficient/intercept, R-square"""
        self.performanceTest = LinearPerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, model=model)

    def _update_binary_performance_plot_roc_curve(self, y_test, y_test_hat, y_test_hat_prob=None,
                                                  model_name=None, display_roc_curve=True):
        """ update performance for classification models & plot roc curve """
        self.performanceTest = BinaryPerformanceSummary(y_test=y_test,
                                                        y_test_hat=y_test_hat,
                                                        y_test_hat_prob=y_test_hat_prob)
        if display_roc_curve:
            self.performanceTest.plot_roc_curve(model_name=model_name)

    def _update_decision_tree_performance_plot_path(self, x_test, y_test, y_test_hat, tree_model, feature_names,
                                                    display_decision_path=True):
        """
        update performance for decision tree & plot decision tree path
        :param x_test: (true) x test set
        :param y_test: (true) y test set
        :param y_test_hat: (pred) y test predicted set
        :param tree_model: tree model object
        :param feature_names: list of feature names
        :param display_decision_path: whether plot separate & combined decision tree paths graph
        """
        self.performanceTest = TreePerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, tree_model=tree_model)

        if display_decision_path:
            output_path = 'outputs/decision_tree_path/'
            # combined tree path
            self.performanceTest.plot_decision_path(features=feature_names, x_test=x_test,
                                                    file_name='{}combined_tree.png'.format(output_path))
            # separate tree paths
            i = 0
            for datapoint in x_test:
                self.performanceTest.plot_decision_path(features=feature_names, x_test=[datapoint],
                                                        file_name='{}tree{}.png'.format(output_path, i))
                i += 1


class MultiClassifiers:
    def __init__(self, df, features, y_name):
        self.df = df
        self.features = features
        self.y_name = y_name
        self.performancesTest = None

    def _update_linear_performances(self, performance_list):
        self.performancesTest = BootstrapLinearPerformanceSummary(performance_list=performance_list)

    def _update_binary_performances_plot_roc_curve(self, performance_list, display_roc_curve):
        self.performancesTest = BootstrapBinaryPerformanceSummary(performance_list=performance_list)
        if display_roc_curve:
            self.performancesTest.bootstrap_roc_curve()

    def _update_decision_tree_performances(self, performance_list):
        self.performancesTest = BootstrapTreePerformanceSummary(performance_list=performance_list)


class LinearReg(Classifier):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run(self, degree_of_polynomial, random_state, test_size=0.2, interaction_only=True,
            lasso=False, ridge=False, alpha=0.1, if_standardize=True):
        """
        run linear regression model
        :param degree_of_polynomial: The degree of the polynomial features.
        :param test_size: size of the test set
        :param random_state: random state number
        :param interaction_only: whether only include interaction term
        :param lasso: whether use Lasso linear regression model
        :param ridge: whether use Ridge linear regression model
        :param alpha: the degree of sparsity of the estimated coefficients for Lasso or Ridge
        :param if_standardize: whether standardize the features
        """

        if if_standardize:
            X = standardize(np.asarray(self.df[self.features]))
        else:
            X = np.asarray(self.df[self.features])
        y = np.asarray(self.df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # add polynomial terms
        poly = PolynomialFeatures(degree_of_polynomial, include_bias=True, interaction_only=interaction_only)
        x_train_poly = poly.fit_transform(x_train)
        x_test_poly = poly.fit_transform(x_test)

        # fit model
        if lasso:
            reg = linear_model.Lasso(alpha=alpha).fit(X=x_train_poly, y=y_train)
        elif ridge:
            reg = linear_model.Ridge(alpha=alpha).fit(X=x_train_poly, y=y_train)
        else:
            reg = linear_model.LinearRegression().fit(X=x_train_poly, y=y_train)

        # prediction
        y_test_hat = reg.predict(x_test_poly)

        # update performance
        self._update_linear_performance(y_test=y_test, y_test_hat=y_test_hat, model=reg)


class MultiLinearReg(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, degree_of_polynomial, test_size=0.2, interaction_only=True,
                 lasso=False, ridge=False, alpha=0.1, if_standardize=True):

        performance_test_list = []
        i = 0
        model = LinearReg(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(degree_of_polynomial=degree_of_polynomial, interaction_only=interaction_only,
                      random_state=i, test_size=test_size, if_standardize=if_standardize,
                      lasso=lasso, ridge=ridge, alpha=alpha)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self._update_linear_performances(performance_list=performance_test_list)


class LogisticReg(Classifier):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run(self, random_state, test_size=0.2, penalty='l2', l1_solver='liblinear',
            display_roc_curve=True, if_standardize=True):
        """
        :param random_state: random state number
        :param l1_solver: solver that handle 'l1' penalty. 'liblinear' good for small dataset, 'sage' good for large set
        :param test_size: size of test sample
        :param penalty: "l1" or "l2", default "l2"
        :param display_roc_curve: whether plot roc curve
        :param if_standardize: whether standardize the features
        """

        if if_standardize:
            X = standardize(np.asarray(self.df[self.features]))
        else:
            X = np.asarray(self.df[self.features])
        y = np.asarray(self.df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # fit model
        solver = 'lbfgs'
        if penalty == 'l1':
            solver = l1_solver
        LR = linear_model.LogisticRegression(penalty=penalty, solver=solver)
        LR.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = LR.predict(x_test)
        y_test_hat_prob = LR.predict_proba(x_test)

        # update model performance attributes
        self._update_binary_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob,
            model_name='Logistic Regression', display_roc_curve=display_roc_curve)


class MultiLogisticReg(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, test_size=0.2, penalty='l2', l1_solver='liblinear',
                 display_roc_curve=True, if_standardize=True):

        performance_test_list = []
        i = 0
        model = LogisticReg(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(random_state=i, test_size=test_size, if_standardize=if_standardize,
                      penalty=penalty, l1_solver=l1_solver, display_roc_curve=False)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self._update_binary_performances_plot_roc_curve(performance_list=performance_test_list,
                                                        display_roc_curve=display_roc_curve)


class DecisionTree(Classifier):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run(self, test_size=0.2, display_decision_path=True):
        X = np.asarray(self.df[self.features])
        y = np.asarray(self.df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rng)

        # fit model
        clf = DecisionTreeRegressor(random_state=24)
        clf.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = clf.predict(x_test)

        # update model performance attributes
        self._update_decision_tree_performance_plot_path(
            x_test=x_test, y_test=y_test, y_test_hat=y_test_hat, feature_names=self.features,
            tree_model=clf, display_decision_path=display_decision_path)


class MultiDecisionTrees(MultiClassifiers):
    def __init__(self, df, features, y_name):
        super().__init__(df, features, y_name)

    def run_many(self, num_bootstraps, test_size=0.2, penalty='l2', l1_solver='liblinear', display_roc_curve=True):

        performance_test_list = []
        i = 0
        model = DecisionTree(df=self.df, features=self.features, y_name=self.y_name)

        while len(performance_test_list) < num_bootstraps:
            model.run(test_size=test_size, display_decision_path=False)
            # append performance
            performance_test_list.append(model.performanceTest)

            i += 1

        self._update_decision_tree_performances(performance_list=performance_test_list)


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
    def __init__(self, y_test, y_test_hat, model):
        super().__init__(y_test, y_test_hat)
        self.r2 = r2_score(y_true=y_test, y_pred=y_test_hat)
        self.mse = mean_squared_error(y_true=y_test, y_pred=y_test_hat)
        self.coefficient = model.coef_
        self.intercept = model.intercept_

    def print(self):
        # print('Coefficients:', self.coefficient)
        # print('Intercept:', self.intercept)
        print('R2:', self.r2)
        print('MSE:', self.mse)


class BootstrapLinearPerformanceSummary(BootstrapPerformanceSummary):
    def __init__(self, performance_list):
        super().__init__(performance_list=performance_list)
        self.statR2 = Stat.SummaryStat(name="R-square", data=[performance.r2 for performance in self.performances])
        self.statMSE = Stat.SummaryStat(name='MSE', data=[performance.mse for performance in self.performances])

    def print(self, decimal=3):
        print('R2:', self.statR2.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('MSE:', self.statMSE.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))


class BinaryPerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, y_test_hat_prob):
        super().__init__(y_test, y_test_hat, y_test_hat_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sen = tp / (tp + fn)
        self.spe = tn / (tn + fp)
        self.fpr, self.tpr, threshold = roc_curve(y_test, y_test_hat_prob[:, 1], drop_intermediate=False)
        self.roc_auc = auc(self.fpr, self.tpr)

    def print(self):
        print("Sensitivity:", self.sen)
        print("Specificity:", self.spe)
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
        print('Sensitivity:', self.statSen.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('Specificity:', self.statSpe.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
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


class TreePerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, tree_model):
        super().__init__(y_test, y_test_hat)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sen = tp / (tp + fn)
        self.spe = tn / (tn + fp)
        self.model = tree_model
        self.fi = tree_model.feature_importances_

    def print(self):
        print("Sensitivity:", self.sen)
        print("Specificity:", self.spe)
        print('Feature Importance:', self.fi)

    def plot_decision_path(self, features, x_test, file_name):
        # ref: https://stackoverflow.com/questions/55878247/how-to-display-the-path-of-a-decision-tree-for-test-samples
        # print graph of decision tree path
        # a visited node is colored in green, all other nodes are white.
        dot_data = export_graphviz(self.model,
                                   out_file=None, feature_names=features, class_names=['not exceed', 'exceed'],
                                   filled=True, rounded=True, special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # empty all nodes, i.e.set color to white and number of samples to zero
        for node in graph.get_node_list():
            if node.get_attributes().get('label') is None:
                continue
            if 'samples = ' in node.get_attributes()['label']:
                labels = node.get_attributes()['label'].split('<br/>')
                for i, label in enumerate(labels):
                    if label.startswith('samples = '):
                        labels[i] = 'samples = 0'
                node.set('label', '<br/>'.join(labels))
                node.set_fillcolor('white')

        samples = x_test
        decision_paths = self.model.decision_path(samples)

        for decision_path in decision_paths:
            for n, node_value in enumerate(decision_path.toarray()[0]):
                if node_value == 0:
                    continue
                node = graph.get_node(str(n))[0]
                node.set_fillcolor('green')
                labels = node.get_attributes()['label'].split('<br/>')
                for i, label in enumerate(labels):
                    if label.startswith('samples = '):
                        labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

                node.set('label', '<br/>'.join(labels))

        filename = file_name
        graph.write_png(filename)


class BootstrapTreePerformanceSummary(BootstrapPerformanceSummary):
    def __init__(self, performance_list):
        super().__init__(performance_list=performance_list)
        self.statSen = Stat.SummaryStat(name="sensitivity", data=[performance.sen for performance in self.performances])
        self.statSpe = Stat.SummaryStat(name="sensitivity", data=[performance.spe for performance in self.performances])

    def print(self, decimal=3):
        print('Sensitivity:', self.statSen.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
        print('Specificity:', self.statSpe.get_formatted_mean_and_interval(deci=decimal, interval_type="p"))
