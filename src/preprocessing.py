from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

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
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline

def preprocessor_pipeline(numerical_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('label', LabelEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor


# Vérifier l'existence de duplications dans le dataset
def check_duplicates(data):
    return data.duplicated().sum()

# Supprimer les duplications dans le dataset
def drop_duplicates(data):
    data.drop_duplicates(inplace=True)
    return data

# Fonction pour visualiser les données manquantes dans le dataset avec un heatmap
def plot_missing_data(data):
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.isnull(), cbar=False)
    plt.title('Missing data in the dataset')
    plt.show()

# Fonction pour enlever les valeurs aberrantes dans le dataset
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

# Fonction pour visualiser les valeurs aberrantes dans le dataset
def plot_outliers(data, column):
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[column])
    plt.title('Outliers in the dataset')
    plt.show()