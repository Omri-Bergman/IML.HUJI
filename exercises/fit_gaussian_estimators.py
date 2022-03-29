from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import pandas as pd

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    # m = 1000
    # samples = np.random.normal(mu, var, m)
    # esti = UnivariateGaussian()
    # # esti.fit(samples)
    # print(esti.mu_, esti.var_)
    # # Question 2 - Empirically showing sample mean is consistent
    # data = []
    # for i in range(10, m + 10, 10):
    #     esti = UnivariateGaussian()
    #     series = esti.fit(samples[:i])
    #     data.append((abs(series.mu_ - mu), i))
    # data = pd.DataFrame(data, columns=['distance', 'number of samples'])
    # fig = px.line(data, x='number of samples', y='distance', height=700,
    #               width=1200, markers=True)
    # fig.update_layout(title = dict(text= 'Absolute distance between the estimated - and true value of the expectation')).show()
    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    # fig = px.scatter(x=samples, y=esti.pdf(samples), height=700,
    #                  width=1000, labels=dict(x='numbers', y='PDF'))
    #
    # fig.update_layout(title=dict(text='Estimation of the PDF')).show()
    a = np.array(
        [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3,
         1, -4, 1, 2, 1,
         -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1,
         0, 3, 5, 0, -2])
    esti = UnivariateGaussian()
    esti.fit(a)
    print(esti.log_likelihood(mu,var,a))


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array(
        [[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    # Question 4 - Draw samples and print fitted model
    m = 1000
    samples = np.random.multivariate_normal(mu, cov, m)
    esti = MultivariateGaussian()
    esti.fit(samples)
    print(esti.mu_)
    print(esti.cov_)
    # print(cov)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)
    mu_values = np.array([[f1, 0, f2, 0] for f1 in f for f2 in f])
    calc = lambda _mu: esti.log_likelihood(_mu, cov, samples)
    lls = np.apply_along_axis(calc, 1, mu_values).reshape(200, 200)
    fig = go.Figure(data=go.Heatmap(x=f, y=f,
                                    z=lls.tolist()),
                    layout=go.Layout(
                        title='Heatmap of the Log Likelihood'))
    fig.update_layout(xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='F3')),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='F1')),
        title=dict(x=0.5, y=0.975, font=dict(size=50)), font=dict(size=19,))
    fig.show()

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    max1,max3 = np.unravel_index(lls.argmax(),lls.shape)
    print(-f[max1].round(4), f[max3].round(4))

if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
