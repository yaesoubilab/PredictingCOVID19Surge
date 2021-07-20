import pandas as pd

from covid_prediction.prediction_models import *

df = pd.read_csv('outputs/prediction_dataset/cleaned_data.csv')

POLYNOMIAL_DEGREE = 2
features = ['Obs: New hospitalizations',
            'Obs: Cumulative vaccination rate', 'Obs: Cumulative hospitalizations',
            'R0s-0']

y_binary = 'If hospitalization threshold passed'
y_continuous = 'Maximum hospitalization rate'

rng = np.random.RandomState(seed=1)

# linear regression
# print('\nLinear regression:')
# Model_linear = LinearReg(df=df, features=features, y_name=y_continuous)
# Model_linear.run(degree_of_polynomial=POLYNOMIAL_DEGREE, random_state=rng, ridge=True)
# Model_linear.performanceTest.print()

print('\nLinear regression bootstrap:')
MultiModel_linear = MultiLinearReg(df=df, features=features, y_name=y_continuous)
MultiModel_linear.run_many(num_bootstraps=50, degree_of_polynomial=POLYNOMIAL_DEGREE, ridge=True, if_standardize=True)
MultiModel_linear.performancesTest.print(decimal=8)

# # logistic regression
# print('\nLogistic regression:')
# Model_logistic = LogisticReg(df=df, features=features, y_name=y_binary)
# Model_logistic.run(penalty='l2', random_state=rng, degree_of_polynomial=POLYNOMIAL_DEGREE, display_roc_curve=False)
# Model_logistic.performanceTest.print()

print('\nLogistic regression bootstrap:')
MultiModel_logistic = MultiLogisticReg(df=df, features=features, y_name=y_binary)
MultiModel_logistic.run_many(num_bootstraps=50, degree_of_polynomial=POLYNOMIAL_DEGREE, if_standardize=False)
MultiModel_logistic.performancesTest.print(decimal=3)

# # decision tree
# print('\nDecision Tree:')
# Model_tree = DecisionTree(df=df, features=features, y_name=y_binary)
# Model_tree.run(display_decision_path=False)
# Model_tree.performanceTest.print()
#
# print('\nDecision Tree bootstrap:')
# MultiModel_tree = MultiDecisionTrees(df=df, features=features, y_name=y_binary)
# MultiModel_tree.run_many(num_bootstraps=10)
# MultiModel_tree.performancesTest.print(decimal=3)



