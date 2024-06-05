from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC

def get_classification_model(model_type='logistic_regression'):
    if model_type == 'logistic_regression':
        return LogisticRegression()
    elif model_type == 'random_forest':
        return RandomForestClassifier()
    elif model_type == 'svm':
        return SVC()
    else:
        raise ValueError("Unsupported model type for classification")

def get_regression_model(model_type='linear_regression'):
    if model_type == 'linear_regression':
        return LinearRegression()
    elif model_type == 'random_forest':
        return RandomForestRegressor()
    else:
        raise ValueError("Unsupported model type for regression")
