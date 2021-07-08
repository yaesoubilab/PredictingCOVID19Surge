import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


# classes to create predictive models

# we can start with logistic regression but one type of the predictive models
# I was hoping to explore is: https://en.wikipedia.org/wiki/Decision_tree_learning

class Classifier:
    def __init__(self, features, y_name):
        self.features = features
        self.y_name = y_name

    def run(self, df, test_size, rng):
        raise NotImplementedError

    def _update_performance_plot_roc_curve(self, y_test, y_test_hat, y_test_hat_prob=None,
                                           model_name=None, display_roc_curve=True):
        """
        :param y_test: actual binary y_test
        :param y_test_hat: predicted binary y_test
        """
        self.performanceTest = PerformanceSummary(y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob)
        if display_roc_curve:
            self.performanceTest.plot_roc_curve(model_name=model_name)


class LogRegression(Classifier):
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
        self._update_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, y_test_hat_prob=y_test_hat_prob,
            model_name='Logistic Regression', display_roc_curve=display_roc_curve)


class DecisionTree(Classifier):
    def __init__(self, features, y_name):
        super().__init__(features, y_name)

    def run(self, df, test_size=0.2, display_roc_curve=False):
        X = np.asarray(df[self.features])
        y = np.asarray(df[self.y_name])

        # split train vs. test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rng)

        # fit model
        DTR = DecisionTreeRegressor()
        DTR.fit(X=x_train, y=y_train)

        # prediction
        y_test_hat = DTR.predict(x_test)

        # update model performance attributes
        self._update_performance_plot_roc_curve(
            y_test=y_test, y_test_hat=y_test_hat, model_name='Decision Tree', display_roc_curve=display_roc_curve)

        # feature importance
        fi = DTR.feature_importances_
        print('feature importance:', fi)


class PerformanceSummary:
    def __init__(self, y_test, y_test_hat, y_test_hat_prob=None):
        """
        :param y_test: list of true ys for model validation
        :param y_test_hat: list of predicted ys (binary)
        """
        self.y_test = y_test
        self.y_test_hat = y_test_hat
        self.y_test_hat_prob = y_test_hat_prob
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_test_hat).ravel()
        self.sensitivity = tp / (tp + fn)
        self.specificity = tn / (tn + fp)
        if y_test_hat_prob is not None:
            self.fpr, self.tpr, threshold = roc_curve(y_test, y_test_hat_prob[:, 1], drop_intermediate=False)
            self.roc_auc = auc(self.fpr, self.tpr)

    def print(self):
        print("Sensitivity:", self.sensitivity)
        print("Specificity:", self.specificity)
        if self.y_test_hat_prob is not None:
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
        if self.y_test_hat_prob is not None:
            plt.text(0.7, 0.1, 'AUC: {}'.format(round(self.roc_auc, 2)))
        plt.show()
