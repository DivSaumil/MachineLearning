import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer
import seaborn as sns



def extract_features(features, path):
    '''
    Read csv file and extract specified features.

    :param features: list of features to extract
    :param path: string path to csv file
    :return: DataFrame with extracted features, indexed by 'Timestamp'
    '''
    try:
        # Read the CSV file
        df = pd.read_csv(path)

        # Check if all features exist in the DataFrame
        missing_features = [feature for feature in features if feature not in df.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the CSV file: {missing_features}")

        # Extract the specified features
        df = df[features]
        return df

    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {path} is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file at {path} could not be parsed.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def show_metrics(predicted, actual):
    '''
    Function to show metrics- RMSE and R2 score and plot predicted vs actual plot
    :param predicted:
    :param actual:
    :return:
    '''
    rmse = np.sqrt(mean_squared_error(predicted, actual))
    r2_scorei = r2_score(predicted, actual)
    print('RMSE:', rmse)
    print('R2 score:', r2_scorei)

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
    # Create a DataFrame with the imputed data and original column names and index
    df_imputed = pd.DataFrame(df_imputed, columns=df.columns, index=df.index)
    return df_imputed
