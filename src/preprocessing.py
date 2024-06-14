from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

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
        ('onehot', LabelEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    return pipeline


def balance_dataset(data_path, target_column, output_path, random_state=42):
    """
    Équilibre le dataset en assurant que chaque classe ait le même nombre de lignes, 
    basé sur la classe ayant le moins de lignes.

    Parameters:
    - data_path (str): Le chemin du fichier CSV à charger.
    - target_column (str): Le nom de la colonne cible pour équilibrer les classes.
    - output_path (str): Le chemin du fichier CSV où sauvegarder le dataset équilibré.
    - random_state (int): La graine aléatoire pour la reproductibilité (par défaut 42).
    """
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Trouver la taille de la plus petite classe
    min_class_size = df[target_column].value_counts().min()
    
    # Échantillonner chaque classe pour avoir un nombre égal de lignes
    df_balanced = df.groupby(target_column).apply(lambda x: x.sample(min_class_size, random_state=random_state)).reset_index(drop=True)
    
    # Sauvegarder le nouveau dataset
    df_balanced.to_csv(output_path, index=False)
    
    # Vérifier la distribution des classes dans le nouvel ensemble de données
    distribution = df_balanced[target_column].value_counts()
    print("Distribution des classes dans le dataset équilibré :")
    print(distribution)
    
    return df_balanced