import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
import seaborn as sns


def extract_features(features, path):
    ''' Read csv file and extract features , where features is a list
    :param features: list of features
    :param path: string path to csv file
    '''
    df = pd.read_csv(path)
    df = df[features]
    return df


def show_metrics(predicted, actual):
    '''
    Function to show metrics- RMSE and R2 score and plot predicted vs actual plot
    :param predicted:
    :param actual:
    :return:
    '''
    rmse = np.sqrt(mean_squared_error(predicted, actual))
    r2_score = r2_score(predicted, actual)
    print('RMSE:', rmse)
    print('R2 score:', r2_score)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=actual, y=predicted)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.show()


def impute_data(df):
    imputer = KNNImputer(n_neighbors=10)
    df_imputed = imputer.fit_transform(df)
    return df_imputed
