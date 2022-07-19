from data_challenge.agoda_cancellation_estimator import *
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import collections

def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    # full_data = pd.read_csv(filename).dropna().drop_duplicates()
    full_data = pd.read_csv(filename).drop_duplicates()
    features = full_data
    labels = full_data["cancellation_datetime"]

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    # np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\datasets\agoda_cancellation_train.csv")
    # nationals_dict = collections.defaultdict(float)
    # counter = 0
    # print ("total num: ",df['original_payment_method'].nunique())
    # step = 1/df['original_payment_method'].nunique()
    # for n in df['original_payment_method']:
    #     if n not in nationals_dict:
    #         nationals_dict[n] = counter
    #         counter += step
    # for k, v in nationals_dict.items():
    #     print(k, v)


    for index, row in df.iterrows():
        df.at[index, 'duration_nights'] = (pd.to_datetime(row.checkout_date) - pd.to_datetime(row.checkin_date)).days
    del df['checkout_date']
    del df['checkin_date']
    for index, row in df.iterrows():
        print(row.duration_nights)




    # print(df.shape[0])
    # # Creating a dataframe with 75%
    # # values of original dataframe
    # part_75 = df.sample(frac=0.75)
    # print(part_75.shape[0])
    # # Creating dataframe with
    # # rest of the 25% values
    # rest_part_25 = df.drop(part_75.index)
    # #
    # # print("\n75% of the given DataFrame:")
    # # print(part_75)
    # #
    # # print("\nrest 25% of the given DataFrame:")
    # print(rest_part_25.shape[0])
    # part_75.to_csv(r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\data_challenge\TRAIN_SET.csv")
    # rest_part_25.to_csv(
    #     r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\data_challenge\TEST_SET.csv")
    # full_data = pd.read_csv(r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\datasets\agoda_cancellation_train.csv").dropna().drop_duplicates()
    # labels = full_data["cancellation_datetime"]




    # x_train, x_test, y_train, y_test = split_train_test(df, cancellation_labels)
    # del x_train[cancellation_labels.name]
    # del x_test[cancellation_labels.name]
    # print(y_train)

    # Fit model over data
    # estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    # evaluate_and_export(estimator, test_X, "id1_id2_id3.csv")
