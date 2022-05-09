import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'],
            dayfirst=True).dropna().drop_duplicates()
    df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
    df = df[(df['Day'] >= 1) & (df['Day'] <= 31)]
    df = df[(df['Temp'] >= - 70)]
    df['DayOfYear'] = df['Date'].dt.day_of_year
    return df.drop('Temp',1), df['Temp']


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df, y = load_data(r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\datasets\City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df_israel = df[df['Country'] == 'Israel'].dropna()
    y_israel = y[df_israel.index].to_numpy()

    df_israel = df_israel.astype({'Year': str})

    fig21 = px.scatter(df_israel, x='DayOfYear', y=y_israel, color='Year',
                       title="Temperture in israel by day of year",labels={
                     'DayOfYear': "Day Of Year",
                     'y': "Temperature",
                 })
    # fig21.show()

    X_israel = pd.concat([df_israel, y[df_israel.index]], axis=1)
    df_israel_month = X_israel.groupby(['Month'])['Temp'].std()
    fig22 = px.bar(df_israel_month, y='Temp',
                   title="Standard deviation of the daily temperatures of each month",labels={
                     'Temp': "Temp STD",
                 })
    # fig22.show()
    # Question 3 - Exploring differences between countries
    X = pd.concat([df, y[df.index]], axis=1)
    country_month_mean = X.groupby(["Country", "Month"], as_index=False). \
        agg(avg_temp=('Temp', np.mean))

    country_month_std = X.groupby(["Country", "Month"]). \
        agg(temp_std=('Temp', np.std))
    fig3 = px.line(country_month_mean, x='Month', y='avg_temp',
                   line_group='Country', color='Country',
                   error_y=country_month_std['temp_std'],
                   title="Average Temperature And STD of Months")
    # fig3.show()


    # Question 4 - Fitting model for different values of `k`


    X_train, X_test, y_train, y_test = split_train_test(df_israel,y[df_israel.index])
    loss_dict = {}
    for k in range(1,11):
        model = PolynomialFitting(k)
        model._fit(X_train['DayOfYear'].to_numpy(),y_train.to_numpy())
        loss_dict[k] = round(model._loss(X_test['DayOfYear'].to_numpy(), y_test.to_numpy()), 2)

    for k, v in loss_dict.items():
        print("for Degree: ", k, " The Loss is: ", v)
    loss_df = pd.DataFrame(loss_dict.items(), columns=['Degree', 'Loss'])
    fig4 = px.bar(loss_df, x='Degree', y="Loss",
                  title="The MSE Loss Over Polynomial Degree")

    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(5)
    model.fit(df_israel['DayOfYear'].to_numpy(), y_israel)
    loss_dict = {}
    for c in ["The Netherlands","South Africa","Jordan"]:
        df_c = df[df['Country'] == c]
        loss_dict[c] = model.loss(df_c['DayOfYear'],  y[df_c.index].to_numpy())
    # fig5 = px.bar(pd.DataFrame(loss_dict.items(), columns=['Country', 'Loss']),
    #           x='Country', y='Loss'
    #           ,
    #           title="Loss Over Countries (Israel With Degree of 5)",
    #           color='Country').show()