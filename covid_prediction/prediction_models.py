import pydotplus
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, r2_score
import matplotlib.pyplot as plt


class Classifier:
    def __init__(self, features, y_name):
        self.features = features
        self.y_name = y_name

    def _update_linear_performance(self, y_test, y_test_hat, model):
        """ update model performance for linear regression model & print coefficient/intercept, R-square"""
        self.performanceTest = LinearPerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, linear_model=model)
        self.performanceTest.print()

    def _update_binary_performance_plot_roc_curve(self, y_test, y_test_hat, y_test_hat_prob=None,
                                                  model_name=None, display_roc_curve=True):
        """ update performance for classification models & plot roc curve """
        self.performanceTest = BinaryPerformanceSummary(y_test=y_test,
                                                        y_test_hat=y_test_hat,
                                                        y_test_hat_prob=y_test_hat_prob)
        self.performanceTest.print()

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
        self.performanceTest.print()

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


class LinearReg(Classifier):
    def __init__(self, features, y_name):
        super().__init__(features, y_name)

    def run(self, df, test_size=0.2):
        X = np.asarray(df[self.features])
        y = np.asarray(df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # fit model
        reg = LinearRegression().fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = reg.predict(x_test)

        # update performance
        self._update_linear_performance(y_test=y_test, y_test_hat=y_test_hat, model=reg)


class LogisticReg(Classifier):
    def __init__(self, features, y_name):
        super().__init__(features, y_name)

    def run(self, df, test_size=0.2, display_roc_curve=True):
        X = np.asarray(df[self.features])
        y = np.asarray(df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # fit model
        LR = LogisticRegression()
        LR.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = LR.predict(x_test)
        y_test_hat_prob = LR.predict_proba(x_test)

        # update model performance attributes
        self._update_binary_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob,
            model_name='Logistic Regression', display_roc_curve=display_roc_curve)


class DecisionTree(Classifier):
    def __init__(self, features, y_name):
        super().__init__(features, y_name)

    def run(self, df, test_size=0.2, display_decision_path=True):
        X = np.asarray(df[self.features])
        y = np.asarray(df[self.y_name])

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


class PerformanceSummary:
    def __init__(self, y_test, y_test_hat, y_test_hat_prob=None):
        """
        :param y_test: list of true ys for model validation
        :param y_test_hat: list of predicted ys (binary)
        """
        self.y_test = y_test
        self.y_test_hat = y_test_hat
        self.y_test_hat_prob = y_test_hat_prob


class LinearPerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, linear_model):
        super().__init__(y_test, y_test_hat)
        self.r2 = r2_score(y_true=y_test, y_pred=y_test_hat)
        self.coefficient = linear_model.coef_
        self.intercept = linear_model.intercept_

    def print(self):
        print('coefficient:', self.coefficient)
        print('intercept:', self.intercept)
        print('R2 score:', self.r2)


class BinaryPerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, y_test_hat_prob):
        super().__init__(y_test, y_test_hat, y_test_hat_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.fpr, self.tpr, threshold = roc_curve(y_test, y_test_hat_prob[:, 1], drop_intermediate=False)
        self.roc_auc = auc(self.fpr, self.tpr)

    def print(self):
        print("Sensitivity:", self.sensitivity)
        print("Specificity:", self.specificity)
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


class TreePerformanceSummary(PerformanceSummary):
    def __init__(self, y_test, y_test_hat, tree_model):
        super().__init__(y_test, y_test_hat)
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        self.model = tree_model
        self.fi = tree_model.feature_importances_

    def print(self):
        print("Sensitivity:", self.sensitivity)
        print("Specificity:", self.specificity)
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
