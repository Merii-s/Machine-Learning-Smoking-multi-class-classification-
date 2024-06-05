import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(data, column):
    sns.countplot(x=column, data=data)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_numeric_distribution(data, column):
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.show()

def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
