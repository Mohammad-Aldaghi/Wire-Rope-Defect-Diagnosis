import pywt
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")


def wavelet_plotter_ct(name: str) -> None:
    """
    plots a simple verion of the countinous wavelet.

    Parameters
    ----------
        name (str): name of the wavelet.

    Returns:
        None
    """

    wavelet = pywt.ContinuousWavelet(name)
    y, x = wavelet.wavefun()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="Red")

    ax.set_title(f"Continuous Wavelet: {name}")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")

    plt.show()


def wavelet_plotter_dt(name: str) -> None:
    """
    Plots a discrete wavelet's scaling and wavelet functions.

    Parameters
    ----------
        name (str): name (str): The name of the discrete wavelet (e.g., "db4", "haar").

    Returns:
        None
    """

    try:
        wavelet = pywt.Wavelet(name)
        phi, psi, x = wavelet.wavefun(level=10)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, psi, color="red")

        ax.set_title(f"Discrete Wavelet: {name}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Amplitude")
        ax.legend()

        plt.show()

    except ValueError:
        print(f"Wavelet '{name}' not found.")


def compare_wavelets_ct(names: list[str]) -> None:
    """
    Plot multiple continuous wavelets on the same figure for comparison.

    Parameters
    ----------
    names : list[str]
        List of wavelet names (e.g. ["morl", "mexh", "gaus1"])
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    for name in names:
        try:
            wavelet = pywt.ContinuousWavelet(name)
            y, x = wavelet.wavefun()
            ax.plot(x, y, label=name)

        except Exception:
            print(f"Wavelet '{name}' not found.")

    ax.set_title("Continuous Wavelet Comparison")
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.legend()

    plt.show()
