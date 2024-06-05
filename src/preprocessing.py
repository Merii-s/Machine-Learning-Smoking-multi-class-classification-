from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def split_data(data, target, test_size=0.2, val_size=0.1):
    X, y = preprocess_data(data, target)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + val_size), stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_pipeline(model, numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline
