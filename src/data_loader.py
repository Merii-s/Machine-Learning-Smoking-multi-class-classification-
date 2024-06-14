import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def save_processed_data(data, output_path):
    data.to_pickle(output_path)

def load_processed_data(input_path):
    return pd.read_pickle(input_path)

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

def reduce_dataset(data_path, target_column, output_path, train_size=100000, random_state=42):
    """
    Réduit le dataset à un nombre spécifié de lignes tout en conservant les proportions des classes.
    
    Parameters:
    - data_path (str): Le chemin du fichier CSV à charger.
    - target_column (str): Le nom de la colonne cible pour le stratified sampling.
    - output_path (str): Le chemin du fichier CSV où sauvegarder le dataset réduit.
    - train_size (int): Le nombre de lignes du dataset réduit (par défaut 100 000).
    - random_state (int): La graine aléatoire pour la reproductibilité (par défaut 42).
    """
    # Charger les données
    df = pd.read_csv(data_path)
    
    # Réduire le dataset à la taille spécifiée en utilisant un échantillonnage stratifié
    df_reduced, _ = train_test_split(df, stratify=df[target_column], train_size=train_size, random_state=random_state)
    
    # Sauvegarder le nouveau dataset
    df_reduced.to_csv(output_path, index=False)
    
    # Vérifier la distribution des classes dans le nouvel ensemble de données
    distribution = df_reduced[target_column].value_counts()
    print("Distribution des classes dans le dataset réduit :")
    print(distribution)
    
    return df_reduced