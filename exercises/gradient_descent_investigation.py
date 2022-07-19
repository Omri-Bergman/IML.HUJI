import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from IMLearn.model_selection import cross_validate
import math
from IMLearn.metrics import misclassification_error
import warnings


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values = []
    weights = []

    def callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for f in [L1, L2]:
        fig = go.Figure()
        for eta in etas:
            module = f(init.copy())
            callback, values, weights = get_gd_state_recorder_callback()
            weights.append(init)
            lr = FixedLR(eta)
            grad = GradientDescent(learning_rate=lr, callback=callback)
            grad.fit(X=None, y=None, f=module)
            # print(values, weights)
            ##EX1
            if eta == .01:
                plot_descent_path(f, np.array(weights),
                                  title=f"{f.__name__} gradient descent using "
                                        f"fixed eta = "
                                        f"{eta}").show(renderer="browser")


            ##Ex3
            fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, name = eta))
            print(f.__name__,"Eta: ", eta,"min loss: ", min(values))
        fig.update_layout(
            {"title": dict(text=f"The norm as a function of the GD iteration with {f.__name__} module"),
             "xaxis_title": "GDs",
             "yaxis_title": "Norm"})
        fig.show(renderer="browser")



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    for f in [L1, L2]:
        fig = go.Figure()
        for gamma in gammas:
            module = f(init.copy())
            callback, values, weights = get_gd_state_recorder_callback()
            weights.append(init)
            lr = ExponentialLR(eta, gamma)
            grad = GradientDescent(learning_rate=lr, callback=callback)
            grad.fit(X=None, y=None, f=module)

            #EX5
            fig.add_trace(go.Scatter(x=list(range(len(values))), y=values, name = gamma))
            print(f.__name__,"Gamma:", gamma ," Min loss:", min(values))

            if gamma == .95:
                plot_descent_path(f, np.array(weights),
                                  title=f"{f.__name__} gradient descent using "
                                        f"fixed gamma = "
                                        f"{gamma}").show(renderer="browser")

        fig.update_layout(
            {"title": dict(text=f"The norm as a function of the GD iteration with {f.__name__} module"),
             "xaxis_title": "GDs",
             "yaxis_title": "Norm"})
        fig.show(renderer="browser")




def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    warnings.filterwarnings('ignore')
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy().reshape(-1), X_test.to_numpy(), y_test.to_numpy().reshape(-1)

    from utils import custom
    c = [custom[0], custom[-1]]

    max_iter = 20000
    lr = 1e-4
    gd = GradientDescent(max_iter=max_iter, learning_rate=FixedLR(lr),callback=
                                                            get_gd_state_recorder_callback()[
                                                                0])
    model = LogisticRegression(solver=gd)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_train)
    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(np.array(y_train), y_prob)

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show(renderer="browser")

    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print("best alpha: ",best_alpha)
    model.alpha_ = best_alpha
    print("best alpha loss: ", model.loss(X_test, y_test))


    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for p in ["l1", "l2"]:
        if p == "l1":
            print("***results for l1***")
        else:
            print("***results for l2***")
        best_l = 0
        best_val = math.inf
        gd = GradientDescent(learning_rate=FixedLR(lr),
                             max_iter=max_iter)
        model = LogisticRegression(include_intercept=False, penalty=p,
                                   solver=gd)
        for l in lambdas:
            model.lam_ = l
            train_score, val_score = cross_validate(model, X_train, y_train, misclassification_error)
            if val_score < best_val:
                best_val = val_score
                best_l = l
        print("Best lambda: ",best_l, "best val: ",best_val)
        gd = GradientDescent(learning_rate=FixedLR(lr),
                             max_iter=max_iter)
        model = LogisticRegression(
            include_intercept=False,
            penalty=p,
            solver=gd,
            lam=best_l
        )
        model.fit(X_train, y_train)
        test_error = misclassification_error(y_test, model.predict(X_test))
        print("test error: ",test_error)


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
