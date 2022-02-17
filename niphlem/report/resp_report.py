"""
This file contains the code to generate the respiration report.

Many of the functionalites used here have been borrowed from Nilearn
(https://nilearn.github.io/)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import string

from html import escape
from os.path import join as opj
from scipy.signal import welch

from niphlem.clean import _transform_filter
from niphlem.events import compute_max_events, correct_anomalies
from .report_general import (validate_signal, validate_outpath,
                             compute_stats, _dataframe_to_html, _plot_to_svg,
                             plot_average_signal, plot_peaks,
                             generate_rate_df, generate_interval_df
                             )
from .html_report import HTMLReport


def make_resp_report(resp_signal,
                     *,
                     fs,
                     delta,
                     peak_rise=0.5,
                     high_pass=0.1,
                     low_pass=0.5,
                     outpath=None,
                     ):
    """
    Generate QC report for respiration data.
    Parameters
    ----------
    resp_signal : array-like of shape (n_physio_samples, ),
        or (n_physio_samples, n_channels).
        Penumatic belt signal.
    fs : float
        Sampling frequency of pneumatic belt recording.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak. The default is 0.5.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 0.1.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 0.5.
    outpath : string, optional
        If provided, Path where report the HTML report,
        averaged filtered signal and corrected peaks will be saved.
        The default is None.
    Returns
    -------
    report : html file
        HTML report.
    output_dict : dict
        Dictionary with the filtered signal and (corrected) peak locations.
    """

    signal = resp_signal.copy()

    signal = validate_signal(signal)

    outpath = validate_outpath(outpath)

    # demean and filter signal
    signal_filt = np.apply_along_axis(_transform_filter,
                                      axis=0,
                                      arr=signal,
                                      high_pass=high_pass,
                                      low_pass=low_pass,
                                      sampling_rate=fs
                                      )
    # Compute average signal across channels for both raw and filter data
    signal = np.mean(signal, axis=1)
    signal_filt = np.mean(signal_filt, axis=1)

    if outpath is not None:
        filepath = opj(outpath, "transformed_signal_resp.txt")
        np.savetxt(filepath, signal_filt)
        print(f"Transformed respiratory signal saved in: {filepath}")

    fig1, peaks, diff_peaks, resp_rate, mean_ipi, median_ipi, \
        stdev_ipi, snr_ipi = plot_transformed_resp(signal,
                                                   signal_filt,
                                                   fs,
                                                   peak_rise,
                                                   delta)

    rr_df = generate_rate_df(fs, diff_peaks, resp_rate)
    ipi_df = generate_interval_df(mean_ipi, median_ipi, stdev_ipi, snr_ipi)

    corrected_peaks, max_indices, min_indices = correct_anomalies(peaks,
                                                                  alpha=0.05,
                                                                  save_name=''
                                                                  )
    # Compute differences between corrected peaks
    corrected_peak_diffs = abs(np.diff(corrected_peaks))

    if outpath is not None:
        filepath = opj(outpath, "peaks_resp.txt")
        np.savetxt(filepath, corrected_peaks)
        print(f"Respiratory peaks saved in: {filepath}")

    fig2, c_resp_rate, c_mean_ipi, c_median_ipi, c_stdev_ipi, c_snr_ipi,\
        c_inst_resp = plot_corrected_resp(signal_filt,
                                          corrected_peaks,
                                          corrected_peak_diffs,
                                          delta, fs)

    corrected_rr_df = generate_rate_df(fs,
                                       corrected_peak_diffs,
                                       c_resp_rate)
    corrected_ipi_df = generate_interval_df(c_mean_ipi, c_median_ipi,
                                            c_stdev_ipi,
                                            c_snr_ipi)

    fig3 = plot_comparison_resp(signal_filt, peaks, diff_peaks, resp_rate,
                                mean_ipi, median_ipi, stdev_ipi,
                                corrected_peaks, corrected_peak_diffs,
                                c_resp_rate, c_mean_ipi, c_median_ipi,
                                c_stdev_ipi, delta, fs)

    # generate html report
    report = _generate_resp_html(fig1, fig2, fig3, rr_df, ipi_df,
                                 max_indices, min_indices,
                                 corrected_rr_df, corrected_ipi_df,
                                 fs,
                                 high_pass,
                                 low_pass,
                                 delta,
                                 peak_rise)

    if outpath is not None:
        filepath = opj(outpath, "resp_qc.html")
        report.save_as_html(filepath)
        print(f"QC report for pneumatic belt signal saved in: {filepath}")

    # Store filtered data and peaks in a dictionary for output
    output_dict = {'filtered_signal': signal_filt,
                   'peaks': corrected_peaks
                   }

    return report, output_dict


def plot_transformed_signal_resp(ax, signal, signal_filt):
    # plots comparison between raw and transformed signal (one panel)
    ax.plot(signal, label="raw signal")
    ax.plot(signal_filt, label="transformed signal")
    # ax.set_xlim([5000, 7000])
    ax.legend()
    return ax


def plot_power_spectrum_resp(ax, signal, signal_filt, fs):
    # plots power spectrum of raw, transformed signal (one panel)
    f, Pxx = welch(signal.flatten(), fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="raw signal")
    f, Pxx = welch(signal_filt, fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="transformed signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectrum")
    ax.set_xlim([0, 20])
    ax.legend()

    return ax


def plot_ipi_hist(ax, diff_peaks):

    ax.hist(diff_peaks, bins=50, density=True)
    ax.set_xlabel("IPI (ms)")
    ax.set_ylabel("Probability density")

    return ax


def plot_inst_resp(ax, fs, diff_peaks):

    inst_resp = (fs/diff_peaks)*60
    ax.plot(inst_resp)
    ax.set_ylim(12, 30)
    # ax.set_xlabel("RR interval number")
    ax.set_ylabel("Breaths per minute")

    return ax, inst_resp


def plot_transformed_resp(signal, signal_filt, fs, peak_rise, delta):

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12))

    x_i = 0
    x_f = 10000
    if signal.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal.shape[0]
    ax1 = axs[0, 0]
    ax1 = plot_transformed_signal_resp(ax1, signal, signal_filt)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f])

    ax2 = axs[0, 1]
    ax2 = plot_power_spectrum_resp(ax2, signal, signal_filt, fs)
    ax2.set_title("B", size=15)

    # Compute peaks
    peaks = compute_max_events(signal_filt,
                               peak_rise=peak_rise,
                               delta=delta)
    diff_peaks = abs(np.diff(peaks))
    # Respiration rate using the difference time between peaks
    resp_rate = np.mean(fs/diff_peaks)*60

    ax3 = axs[1, 0]
    ax3 = plot_peaks(ax3, signal_filt, peaks)
    ax3.set_title("C", size=15)
    ax3.set_xlim([x_i, x_f])

    # Compute signal around peaks
    ax4 = axs[1, 1]
    ax4 = plot_average_signal(ax4, peaks, delta, signal_filt)
    ax4.set_title("D: Respiration rate = %.2f bpm" % resp_rate, size=15)

    # Compute mean, median, stdev, snr of IPI
    mean_ipi, median_ipi, stdev_ipi, snr_ipi = compute_stats(diff_peaks)

    # Compute peaks and plot histogram of IPI
    ax5 = axs[2, 0]
    ax5 = plot_ipi_hist(ax5, diff_peaks)
    ax5.set_title("E: IPI mean = %.2f, "
                  "median = %.2f, "
                  "stdev = %.2f" % (mean_ipi, median_ipi, stdev_ipi),
                  size=13)

    # Compute and plot instantaneous resp rate
    ax6 = axs[2, 1]
    ax6, inst_resp = plot_inst_resp(ax6, fs, diff_peaks)
    ax6.set_title("F: Instantaneous respiration rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig, peaks, diff_peaks, resp_rate, mean_ipi, median_ipi, stdev_ipi, snr_ipi


def plot_corrected_resp(signal_filt, peaks, diff_peaks, delta, fs):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

    x_i = 0
    x_f = 10000
    if signal_filt.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal_filt.shape[0]

    ax1 = axs[0, 0]
    ax1 = plot_peaks(ax1, signal_filt, peaks)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f])

    # Respiration rate using the difference time between peaks
    resp_rate = np.mean(fs/diff_peaks)*60

    # Compute signal around peaks
    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, peaks, delta, signal_filt)
    ax2.set_title("B: Corrected respiration rate = %.2f bpm" % resp_rate, size=15)

    # Compute mean, median, stdev, snr of IPI
    mean_ipi, median_ipi, stdev_ipi, snr_ipi = compute_stats(diff_peaks)

    # Compute peaks and plot histogram of IPI
    ax3 = axs[1, 0]
    ax3 = plot_ipi_hist(ax3, diff_peaks)
    ax3.set_title("C: IPI mean = %.2f, "
                  "median = %.2f,stdev = %.2f" % (mean_ipi,
                                                  median_ipi,
                                                  stdev_ipi),
                  size=15)

    # Compute and plot instantaneous respiration rate
    ax4 = axs[1, 1]
    ax4, inst_resp = plot_inst_resp(ax4, fs, diff_peaks)
    ax4.set_title("D: Corrected Instantaneous respiration rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig, resp_rate, mean_ipi, median_ipi, stdev_ipi, snr_ipi, inst_resp


def plot_comparison_resp(signal_filt, peaks, diff_peaks, resp_rate,
                         mean_ipi, median_ipi, stddev_ipi,
                         corrected_peaks, corrected_diff_peaks2,
                         corrected_resp_rate, c_mean_ipi, c_median_ipi,
                         c_stdev_ipi, delta, fs):

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12))

    ax1 = axs[0, 0]
    ax1 = plot_average_signal(ax1, peaks, delta, signal_filt)
    ax1.set_title("A: Respiration rate = %.2f bpm" % resp_rate, size=15)

    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, corrected_peaks, delta, signal_filt)
    ax2.set_title("B: Corrected respiration rate = %.2f bpm" % corrected_resp_rate, size=15)

    # Plot histogram of IPI
    ax3 = axs[1, 0]
    ax3 = plot_ipi_hist(ax3, diff_peaks)
    ax3.set_title("C: IPI mean = %.2f, "
                  "med = %.2f, "
                  "stdev = %.2f" % (mean_ipi, median_ipi, stddev_ipi),
                  size=15)

    ax4 = axs[1, 1]
    ax4 = plot_ipi_hist(ax4, corrected_diff_peaks2)
    ax4.set_title("D: IPI mean = %.2f, "
                  "med = %.2f, "
                  "stdev = %.2f" % (c_mean_ipi,
                                    c_median_ipi,
                                    c_stdev_ipi),
                  size=15)

    # Plot instantaneous respiration rate
    ax5 = axs[2, 0]
    ax5, inst_resp = plot_inst_resp(ax5, fs, diff_peaks)
    ax5.set_title("E: Instantaneous respiration rate", size=15)

    ax6 = axs[2, 1]
    ax6, c_inst_resp = plot_inst_resp(ax6, fs, corrected_diff_peaks2)
    ax6.set_title("F: Corrected instantaneous respiration rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig


def _generate_resp_html(fig1, fig2, fig3, rr_df, ipi_df,
                        max_indices, min_indices,
                        corrected_rr_df, corrected_ipi_df,
                        fs, high_pass, low_pass, delta, peak_rise
                        ):
    """ Returns HTMLReport object for a QC report which shows
    results of signal processing and anomaly correction.
    The object can be opened in a browser, displayed in a notebook,
    or saved to disk as a standalone HTML file.
    Examples
    --------
    report = make_glm_report(model, contrasts)
    report.open_in_browser()
    report.save_as_html(destination_path)
    Parameters
    ----------
    Returns
    -------
    report_text : HTMLReport Object
        Contains the HTML code for the GLM Report.
    """

    HTML_TEMPLATE_ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

    html_head_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                           'report_head.html')

    html_body_template_path = os.path.join(HTML_TEMPLATE_ROOT_PATH,
                                           'report_body_resp.html')

    with open(html_head_template_path) as html_head_file_obj:
        html_head_template_text = html_head_file_obj.read()
    report_head_template = string.Template(html_head_template_text)

    with open(html_body_template_path) as html_body_file_obj:
        html_body_template_text = html_body_file_obj.read()
    report_body_template = string.Template(html_body_template_text)

    page_title = 'niphlem'
    page_heading = ('niphlem: pneumatic belt signal processing and '
                    'peak detection QC report')

    fig1_html = _plot_to_svg(fig1)
    fig2_html = _plot_to_svg(fig2)
    fig3_html = _plot_to_svg(fig3)

    rr_html = _dataframe_to_html(rr_df,
                                 precision=2,
                                 header=True,
                                 sparsify=False
                                 )

    ipi_html = _dataframe_to_html(ipi_df,
                                  precision=2,
                                  header=True,
                                  sparsify=False,
                                  )

    corrected_rr_html = _dataframe_to_html(corrected_rr_df,
                                           precision=2,
                                           header=True,
                                           sparsify=False,
                                           )

    corrected_ipi_html = _dataframe_to_html(corrected_ipi_df,
                                            precision=2,
                                            header=True,
                                            sparsify=False,
                                            )

    report_values_head = {'page_title': escape(page_title)}
    report_values_body = {'page_heading': page_heading,
                          'fs': fs,
                          'low_cut': high_pass,
                          'high_cut': low_pass,
                          'delta': delta,
                          'peak_rise': peak_rise,
                          'fig1_html': fig1_html,
                          'rr_html': rr_html,
                          'ipi_html': ipi_html,
                          'max_indices': max_indices,
                          'min_indices': min_indices,
                          'fig2_html': fig2_html,
                          'corrected_rr_html': corrected_rr_html,
                          'corrected_ipi_html': corrected_ipi_html,
                          'fig3_html': fig3_html
                          }

    report_text_body = report_body_template.safe_substitute(**report_values_body)
    report_text = HTMLReport(body=report_text_body,
                             head_tpl=report_head_template,
                             head_values=report_values_head)

    return report_text
