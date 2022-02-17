"""
This file contains some general code to aid the generation of the reports.

Some of the code used here has been borrowed from Nilearn
(https://nilearn.github.io/)
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import io
import urllib


def compute_stats(x):

    mean_x = np.mean(x)
    median_x = np.median(x)
    stdev_x = np.std(x)
    snr_x = mean_x/stdev_x

    return mean_x, median_x, stdev_x, snr_x


def plot_peaks(ax, signal_filt, peaks):

    ax.plot(signal_filt)

    ax.scatter(peaks.astype(int),
               signal_filt[peaks.astype(int)],
               c="red", marker="x",
               s=100)
    ax.set_xlabel("Time (s)")

    return ax


def plot_average_signal(ax, peaks, delta, signal_filt):

    sign_peaks = []
    for pk in peaks.astype(int):
        i_0 = pk-delta
        i_f = pk+delta
        if i_0 < 0:
            continue
        if i_f > len(signal_filt):
            continue
        sign_peaks.append(signal_filt[i_0:i_f])

    ax.plot(np.mean(np.array(sign_peaks), axis=0))
    ax.errorbar(x=np.arange(2*delta),
                y=np.mean(np.array(sign_peaks), axis=0),
                yerr=np.std(np.array(sign_peaks), axis=0),
                alpha=0.5
                )

    return ax


def generate_rate_df(fs, diff_peaks, signal_rate):

    # generate table of statistics for signal rate
    # (either resp rate or heart rate)
    # already calculated: signal_rate (mean)

    all_rate = (fs/diff_peaks)*60

    # create 95% confidence interval
    lower_bound, upper_bound = st.t.interval(alpha=0.95,
                                             df=len(all_rate)-1,
                                             loc=signal_rate,
                                             scale=st.sem(all_rate)
                                             )

    rate_dict = {'mean': [signal_rate],
                 '95% CI (lower bound)': [lower_bound],
                 '95% CI (upper bound)': [upper_bound]
                 }
    rate_df = pd.DataFrame(data=rate_dict)

    return rate_df


def generate_interval_df(mean_ipi, median_ipi, stdev_ipi, snr_ipi):

    # generate table of statistics for IPI
    # already calculated: mean_ipi, median_ipi, stdev_ipi, snr_ipi

    ipi_dict = {'mean': [mean_ipi], 'median': [median_ipi],
                'standard deviation': [stdev_ipi],
                'SNR': [snr_ipi]
                }
    ipi_df = pd.DataFrame(data=ipi_dict)

    return ipi_df


def _dataframe_to_html(df, precision, **kwargs):
    """Makes HTML table from provided dataframe.
    Removes HTML5 non-compliant attributes (ex: `border`).
    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to be converted into HTML table.
    precision : int
        The display precision for float values in the table.
    **kwargs : keyworded arguments
        Supplies keyworded arguments for func: pandas.Dataframe.to_html()
    Returns
    -------
    html_table : String
        Code for HTML table.
    """
    with pd.option_context('display.precision', precision):
        html_table = df.to_html(**kwargs)
    html_table = html_table.replace('border="1" ', '')
    return html_table


def figure_to_svg_bytes(fig):
    with io.BytesIO() as io_buffer:
        fig.savefig(
            io_buffer, format="svg", facecolor="white", edgecolor="white"
        )
        return io_buffer.getvalue()


def figure_to_svg_quoted(fig):
    return urllib.parse.quote(figure_to_svg_bytes(fig).decode("utf-8"))


def _plot_to_svg(plot):
    """Creates an SVG image as a data URL
    from a Matplotlib Axes or Figure object.
    Parameters
    ----------
    plot : Matplotlib Axes or Figure object
        Contains the plot information.
    Returns
    -------
    url_plot_svg : String
        SVG Image Data URL.
    """
    try:
        return figure_to_svg_quoted(plot)
    except AttributeError:
        return figure_to_svg_quoted(plot.figure)


def validate_signal(signal, ground=None):
    """
    This function validates the input data for the report generattion.
    Basically it returns a 2D array, even in the case of just one channel.
    Later during the reports generation, signal is collapsed across channels
    by averaging them. A ground argument is added for substraction (e.g. ECG).
    """

    signal = np.squeeze(signal)

    if signal.ndim > 2:
        raise ValueError("Supplied signal should be at most 2D")
    if signal.ndim == 2:
        if ground is not None:
            n_ch = signal.shape[1]
            # Substract ground from signal
            ground = int(ground)
            signal -= signal[:, [ground]]
            signal = signal[:, np.arange(n_ch) != ground]
    else:
        signal = signal[:, None]

    return signal


def validate_outpath(outpath):
    "Validates outpath."
    from pathlib import Path

    if outpath is not None:
        try:
            outpath = Path(outpath)
        except TypeError:
            raise ValueError("outpath should be a string")

        outpath.mkdir(exist_ok=True, parents=True)
        outpath = outpath.absolute().as_posix()
    return outpath
