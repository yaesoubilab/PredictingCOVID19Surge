import pandas as pd
import pydotplus
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.tree import export_graphviz

from covid_prediction.prediction_models import DecisionTree

# load dataset
pima = pd.read_csv("pima-indians-diabetes.csv")

# print(pima.head())

# split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
X = pima[feature_cols]  # Features
y = pima.Outcome  # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

# Create Decision Tree classifier object
clf = DecisionTreeClassifier(random_state=1)

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ------------ optimizing
# Create Decision Tree classifier object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# ----------- figure ------------
dot_data = export_graphviz(clf,
                           out_file=None, filled=True, rounded=True, proportion=True, impurity=False,
                           special_characters=True, feature_names=feature_cols, class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('diabetes.png')


# ----------- testing DecisionTree class -----------
dt = DecisionTree(df=pima, features=feature_cols, y_name='Outcome')
dt.run(test_size=0.3, criterion="entropy", max_depth=3, save_decision_path_filename='diabetes2.png')
dt.performanceTest.print()

