from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = "browser"

PRICE_LABEL = "price"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """


    df_origin = pd.read_csv(filename).dropna().drop_duplicates()
    df = df_origin
    df.drop(df[df.price <= 0].index, inplace=True) ##delete sampels where price <= 0

    del df['id']
    del df['lat']
    del df['long']
    del df['date']
    'condition Values'
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    for i in ["sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[i] > 0]
    for i in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[i] >= 0]
    df['last_build'] = df.apply(lambda x: x['yr_renovated'] if x['yr_renovated'] >=
                     x['yr_built'] else x['yr_built'], axis=1)
    df = pd.get_dummies(df, prefix='last_build', columns=['last_build'])
    df['is_renovated'] = np.where(df.yr_renovated > 0, 1, 0) ## add bool feature if the house renovated
    y = df[PRICE_LABEL]
    del df[PRICE_LABEL]
    return df, y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """


    for i in X:
        pearson = np.cov(X[i], y)[0, 1] / (np.std(X[i]) * np.std(y))
        fig = px.scatter(pd.DataFrame({'x': X[i], 'y': y}), x="x", y="y",
                         trendline="ols",
                         title=f"Correlation Between {i} Values and Response Pearson Correlation {pearson}",
                         labels={"x": f"{i} Values", "y": "Response Values"})
        fig.write_image(output_path + i + ".jpeg")
        # fig.update_layout(title=dict(text='Estimation of the PDF')).show()



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df , y= load_data(r"C:\Users\omrib\PycharmProjects\IML\IML.HUJI\datasets\house_prices.csv")


    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(df, y, "C:\שולחן העבודה\שנה ב\סמסטר ב\IML\ex2\out")

    # # Question 3 - Split samples into training- and testing sets.
    X_train, X_test, y_train, y_test = split_train_test(df,y)
    # print(X_tr['zipcode'].min(), y_tr)
   #
    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    loss_mean = []
    loss_var = []
    for p in range(10,101):
        loss = []
        for i in range(10):
           X = X_train.sample(frac=p/100)
           y = y_train[X.index].to_numpy()
           X = X.to_numpy()

           model = LinearRegression().fit(X,y)
           loss.append(model._loss(X_test.to_numpy(), y_test.to_numpy()))


        loss_mean.append(np.mean(loss))
        loss_var.append(np.std(loss))

    loss_mean = np.array(loss_mean)
    loss_var = np.array(loss_var)

    a = np.arange(10, 101)
    fig4 = go.Figure(
        [go.Scatter(x=a, y=loss_mean, mode='markers+lines', name="Mean Loss"),
         go.Scatter(x=a, y=(loss_mean - 2 * loss_var), fill='tonexty',
                    mode="lines",
                    line=dict(color="lightgrey"), showlegend=False),
         go.Scatter(x=a, y=loss_mean + 2 * loss_var, fill='tonexty',
                    mode="lines",
                    line=dict(color="lightgrey"), showlegend=False)]
        , layout=go.Layout(
            title="Mean and Variance of the loss by the size of the Train Set",
            xaxis_title="Prencetage of train set",
            yaxis_title="MSE"))
    fig4.show()