import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def save_processed_data(data, output_path):
    data.to_pickle(output_path)

def load_processed_data(input_path):
    return pd.read_pickle(input_path)
