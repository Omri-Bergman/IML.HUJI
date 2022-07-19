import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import warnings
from IPython.display import display
from data_challenge.Amen_Model import lr
warnings.filterwarnings('ignore')


def get_data(path):
    #Read and load data
    df = pd.read_csv(path)
    df.drop_duplicates(inplace= False)# will keep first row and others consider as duplicate
    # display(df)
    null = 100 * (df.isna().sum() / df.shape[0])
    # print(null)
    drop_lost = ['hotel_brand_code','hotel_chain_code','request_nonesmoke','request_latecheckin',
                 'request_highfloor', 'request_largebed', 'request_twinbeds', 'request_airport','request_earlycheckin','Unnamed: 0']
    df.drop(drop_lost, axis=1, inplace=True) # delete coluoms with high NULL rate
    # print(df.index)

    print("******************")
    df['cancellation_datetime'] = df['cancellation_datetime'].replace(np.nan, 0).astype(bool).astype(int)

    guest_city = df[df['cancellation_datetime'] == 1][
        'guest_nationality_country_name'].value_counts().reset_index()
    guest_city.columns = ['guest_nationality_country_name', 'No of guests']
    # print(guest_city)
    # correlation = df.corr()['cancellation_datetime'].abs().sort_values(ascending=False)
    # print(correlation)
    useless_col = ['hotel_area_code' ,'h_customer_id', 'h_booking_id', 'no_of_extra_bed', 'is_first_booking']
    df.drop(useless_col, axis=1, inplace=True)
    correlation = df.corr()['cancellation_datetime'].abs().sort_values(ascending=False)
    y = df['cancellation_datetime']
    df = pd.get_dummies(df, prefix='hotel_country_code_', columns=['hotel_country_code'])
    df = pd.get_dummies(df, prefix='accommadation_type_name_',columns=['accommadation_type_name'])
    df = pd.get_dummies(df, prefix='charge_option_',columns=['charge_option'])
    df = pd.get_dummies(df, prefix='customer_nationality_',columns=['customer_nationality'])
    df = pd.get_dummies(df, prefix='guest_nationality_country_name_',columns=['guest_nationality_country_name'])
    df = pd.get_dummies(df, prefix='origin_country_code_',columns=['origin_country_code'])
    df = pd.get_dummies(df, prefix='language_',columns=['language'])
    df = pd.get_dummies(df, prefix='original_payment_method_',columns=['original_payment_method'])
    df = pd.get_dummies(df, prefix='original_payment_type_', columns=['original_payment_type'])
    df = pd.get_dummies(df, prefix='original_payment_currency_', columns=['original_payment_currency'])
    df = pd.get_dummies(df, prefix='cancellation_policy_code_', columns=['cancellation_policy_code'])
    correlation = df.corr()['cancellation_datetime'].abs().sort_values(ascending=False)
    # print(correlation)
    for index, row in correlation.iteritems():

        print (index, row)
    temp_drop_labels_dates = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date']
    df.drop(temp_drop_labels_dates, axis=1,
            inplace=True)  # delete coluoms with high NULL rate
    # print(df.columns.values)
    return df.drop(['cancellation_datetime'], axis=1), y
if __name__ == '__main__':
    X_train, y_train = get_data(r"C:\שולחן העבודה\שנה ב\סמסטר ב\IML\Challenge\Data\TRAIN_SET.csv")
    # X_test, y_test = get_data(r"C:\שולחן העבודה\שנה ב\סמסטר ב\IML\Challenge\Data\TEST_SET.csv")
    # common_cols = [col for col in set(X_train.columns).intersection(X_test.columns)]
    # X_test = X_test[common_cols]
    # X_train = X_train[common_cols]
    # print(y_test)
    # print("X_train: ",X_train.shape)
    # print("y_train: ", y_train.shape)
    # print("X_test: ", X_test.shape)
    # print("y_test: ", y_test.shape)
    # lr(X_train, y_train,X_test, y_test)