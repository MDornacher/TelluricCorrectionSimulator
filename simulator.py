from matplotlib import pyplot as plt
import numpy as np


def guassian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def synth_line(x: np.ndarray, mu: float, sigma: float, strength: float, noise: float):
    return 1 - guassian(x, mu, sigma) * strength + np.random.normal(0, noise, len(x))


def correction_plot(
    n: int,
    fit_mu: float,
    fit_sigma: float,
    fit_strength: float,
    line_strength: float,
    noise: float,
    width: float,
):
    xs = np.linspace(-width, width, n)
    ys = synth_line(xs, 0.0, 1.0, line_strength, noise)
    fit_values = synth_line(xs, fit_mu, fit_sigma, fit_strength, 0)
    corrected_values = ys / fit_values

    plt.plot(xs, ys, "k-", label="Feature", drawstyle="steps-mid")
    plt.plot(xs, fit_values, color="C3", label="Fit", drawstyle="steps-mid")
    plt.plot(xs, corrected_values, color="C0", label="Correction", drawstyle="steps-mid")
    plt.grid(True)
    plt.legend(loc=4)
    plt.xlim(-width, width)
    plt.ylim(0, 1.1 * max(ys.max(), fit_values.max(), corrected_values.max()))
    plt.show()
