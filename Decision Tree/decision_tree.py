import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import graphviz
from argparse import ArgumentParser
from enum import Enum
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algorithm", type=str, default="decision_tree", required=False)
parser.add_argument("--generate_tree", action="store_true")
parser.add_argument("--criterion", type=str, default="gini", required=False)
parser.add_argument("--seed", type=int, default=None, required=False)
parser.add_argument("--test_size", type=float, default=0.2, required=False)
parser.add_argument("--cv_enabled", action="store_true")
parser.add_argument("--obtain_most_significant", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--repeat", type=int, default=1, required=False)
args = parser.parse_args()

# Choose algorithm
class Algorithm(Enum):
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"

algorithm = Algorithm(args.algorithm)

# Get data from file
def get_data():
    repo = fetch_ucirepo(id=17)
    X = repo.data.features
    y = repo.data.targets["Diagnosis"]
    return X, y

# Split data into training and testing
def split_data(X, y):
    X_train, X_test, y_train, y_test=  train_test_split(X, y, test_size=args.test_size)
    return X_train, X_test, y_train, y_test

def decision_tree_classifier(**kwargs):
    return DecisionTreeClassifier(criterion=args.criterion, **kwargs)

def random_forest_classifier(**kwargs):
    return RandomForestClassifier(**kwargs)

def get_classifier(**kwargs):
    if algorithm == Algorithm.DECISION_TREE:
        return decision_tree_classifier(**kwargs)
    elif algorithm == Algorithm.RANDOM_FOREST:
        return random_forest_classifier(**kwargs)

# Do cross validation for max_depth
def cross_validation(classifier, X_train, y_train):
    if algorithm == Algorithm.RANDOM_FOREST:
        param_grid = {'max_depth': range(1, 10), 'n_estimators': range(1, 40)}
    else:
        param_grid = {'max_depth': range(1, 10)}
    grid_search = GridSearchCV(classifier, param_grid)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def get_accuracy(classifier, X_test, y_test):
    return classifier.score(X_test, y_test)

def get_tree(classifier):
    if algorithm == Algorithm.RANDOM_FOREST:
        return classifier.estimators_[0]
    else:
        return classifier

def show_tree(tree, X, y):
    dot_data = export_graphviz(
        tree, 
        out_file=None, 
        feature_names=X.columns,  
        class_names=np.unique(y).astype(str),
        filled=True, 
        rounded=True,  
        special_characters=True
    )  
    graph = graphviz.Source(dot_data)  
    graph.render("decision_tree")
    graph.view()

def get_most_significant_features(classifier):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    return indices

def evaluate_linear_model(X_train, X_test, y_train, y_test, selected_features):
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_selected, y_train)
    y_pred_train = model.predict(X_train_selected)
    y_pred_test = model.predict(X_test_selected)
    return accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)

def evaluate_most_significant_features(X_train, X_test, y_train, y_test, indices):
    feature_count = [5, 10, 15, 20]
    for n_features in feature_count:
        selected_features = indices[:n_features]
        train_acc, test_acc = evaluate_linear_model(X_train, X_test, y_train, y_test, selected_features)
        print(f"Top {n_features} Features - Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

if args.seed:
    np.random.seed(args.seed)

# Main
data = get_data()
X_train, X_test, y_train, y_test = split_data(*data)
best_estimator = {}

if args.cv_enabled:
    classifier = get_classifier()
    best_estimator = cross_validation(classifier, X_train, y_train)
    if algorithm == Algorithm.RANDOM_FOREST:
        best_estimator = {
            "max_depth": best_estimator.get_params()["max_depth"],
            "n_estimators": best_estimator.get_params()["n_estimators"]
        }
    else:
        best_estimator = {
            "max_depth": best_estimator.get_params()["max_depth"]
        }
    print(best_estimator)

test_accuracies = []
train_accuracies = []
for _ in range(args.repeat):
    classifier = get_classifier(**best_estimator)
    classifier.fit(X_train, y_train)
    train_accuracy = get_accuracy(classifier, X_train, y_train)
    test_accuracy = get_accuracy(classifier, X_test, y_test)
    test_accuracies.append(test_accuracy)
    train_accuracies.append(train_accuracy)

train_accuracy = np.mean(train_accuracies)
test_accuracy = np.mean(test_accuracies)
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

if args.generate_tree:
    tree = get_tree(classifier)
    show_tree(tree, X_train, y_train)

if args.obtain_most_significant:
    indices = get_most_significant_features(classifier)
    evaluate_most_significant_features(X_train, X_test, y_train, y_test, indices)
    
if args.plot:
    n_estimators = range(1, 40)
    train_accuracies = []
    test_accuracies = []
    for n in n_estimators:
        classifier = random_forest_classifier(n_estimators=n)
        classifier.fit(X_train, y_train)
        train_accuracy = get_accuracy(classifier, X_train, y_train)
        test_accuracy = get_accuracy(classifier, X_test, y_test)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
    
    plt.plot(n_estimators, train_accuracies, label="Train Accuracy")
    plt.plot(n_estimators, test_accuracies, label="Test Accuracy")
    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    