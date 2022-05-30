import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    size = 250
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(wl=DecisionStump, iterations=n_learners)
    model.fit(train_X, train_y)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.linspace(1, n_learners, n_learners),
                y=list(map(
                    lambda n: model.partial_loss(train_X, train_y,
                                                 int(n)),
                    np.linspace(1, n_learners, n_learners))),
                mode='markers+lines',
                name="training data",
                marker=dict(size=5, opacity=0.6),
                line=dict(width=3)
            ),
            go.Scatter(
                x=np.linspace(1, n_learners, n_learners),
                y=list(map(
                    lambda n: model.partial_loss(test_X, test_y,
                                                 int(n)),
                    np.linspace(1, n_learners, n_learners))),
                mode='markers+lines',
                name="test data",
                marker=dict(size=5, opacity=0.6),
                line=dict(width=3)
            )
        ],
        layout=go.Layout(
            title=f"test error with 0.4 noise.",
            xaxis_title={'text': "number of learners"},
            yaxis_title={'text': "loss"}
        )
    )
    fig.show()
    # exit()
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    model_names = [str(t) + 'classifiers' for t in T]
    fig2 = make_subplots(rows=2, cols=2,
                         subplot_titles=[rf"$\textbf{{{m}}} $"r" learners"
                                        for m in T],
                         horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        f = lambda data: model.partial_predict(data, t)
        fig2.add_traces(
            [decision_surface(f, lims[0], lims[1], showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y,
                                    colorscale=[custom[0], custom[-1]],
                                    line=dict(color="black", width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(
        title="Fitted ensemble predictions up to number of iterations",
        margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    # fig2.show()

    # Question 3: Decision surface of best performing ensemble
    losses = [model.partial_loss(test_X, test_y, t) for t in range(size)]
    best_size = np.argmin(losses)
    fig = go.Figure()
    pred_f = lambda data: model.partial_predict(data, best_size)
    acc = np.sum(model.partial_predict(test_X,best_size) == test_y) / test_y.shape[0]
    fig.add_traces([decision_surface(pred_f,
                                     lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                               mode="markers", showlegend=False,
                               marker=dict(color=test_y,
                                           colorscale=[custom[0],
                                                       custom[-1]],
                                           line=dict(color="black",
                                                     width=1)))])
    fig.update_layout(title="Best ensemble size predict " + str(best_size) + " Accuracy " + str(acc),
                      margin=dict(t=100)).update_xaxes(visible=False). \
        update_yaxes(visible=False)
    # fig.show()
    # Question 4: Decision surface with weighted samples
    D_normal = 5 * model.D_ / np.max(model.D_)
    fig = go.Figure(
        [decision_surface(model.predict, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
                    showlegend=False,
                    marker=dict(color=test_y,
                                symbol='diamond',
                                size=D_normal,
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))],
        layout=go.Layout(
            title=rf"$\textbf{{Decision boundry of last iteration "
                            rf"with points size proportianl to their ADA " 
                            rf"weight and normalized}}$",
            font_size=15
        )
    )
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.4)
