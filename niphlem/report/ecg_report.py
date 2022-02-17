"""
This file contains the code to generate the ECG report.

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
                             compute_stats, _dataframe_to_html,
                             _plot_to_svg, plot_average_signal, plot_peaks,
                             generate_rate_df, generate_interval_df
                             )

from .html_report import HTMLReport


def make_ecg_report(ecg_signal,
                    *,
                    fs,
                    delta,
                    peak_rise=0.75,
                    ground=None,
                    high_pass=0.6,
                    low_pass=5.0,
                    outpath=None,
                    ):
    """
    Generate QC report for ECG data.
    Parameters
    ----------
    ecg_signal : array-like of shape (n_physio_samples, ),
        or (n_physio_samples, n_channels).
        ECG Signal, where each column corresponds to a recording.
    fs : float
        Sampling frequency of ECG recording.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak. The default is 0.75.
    ground : integer, optional
        Column in the input signal to be considered as a ground channel.
        This signal will be then substracted from the other channels.
        The default is None.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 0.6.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 5.0.
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

    signal = ecg_signal.copy()

    signal = validate_signal(signal, ground=ground)

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
        filepath = opj(outpath, "transformed_signal_ecg.txt")
        np.savetxt(filepath, signal_filt)
        print(f"Transformed ECG signal saved in: {filepath}")

    fig1, peaks, diff_peaks, heart_rate, mean_RR, median_RR, \
        stdev_RR, snr_RR = plot_filtered_data(signal,
                                              signal_filt,
                                              fs,
                                              peak_rise,
                                              delta)

    hr_df = generate_rate_df(fs, diff_peaks, heart_rate)
    rr_df = generate_interval_df(mean_RR, median_RR, stdev_RR, snr_RR)

    corrected_peaks, max_indices, min_indices = correct_anomalies(peaks)
    # Compute differences between corrected peaks
    corrected_peak_diffs = abs(np.diff(corrected_peaks))

    if outpath is not None:
        filepath = opj(outpath, "corrected_peaks_ecg.txt")
        np.savetxt(filepath, corrected_peaks)
        print(f"ECG peaks saved in: {filepath}")

    fig2, c_heart_rate, c_mean_RR, c_median_RR, c_stdev_RR, c_snr_RR,\
        c_inst_hr = plot_corrected_ecg(signal_filt,
                                       corrected_peaks,
                                       corrected_peak_diffs,
                                       delta, fs)

    corrected_hr_df = generate_rate_df(fs,
                                       corrected_peak_diffs,
                                       c_heart_rate)
    corrected_rr_df = generate_interval_df(c_mean_RR, c_median_RR, c_stdev_RR,
                                           c_snr_RR)

    fig3 = plot_comparison_ecg(signal_filt, peaks, diff_peaks, heart_rate,
                               mean_RR, median_RR, stdev_RR,
                               corrected_peaks, corrected_peak_diffs,
                               c_heart_rate, c_mean_RR, c_median_RR,
                               c_stdev_RR, delta, fs)

    # generate html report
    report = _generate_ecg_html(fig1, fig2, fig3, hr_df, rr_df,
                                max_indices, min_indices,
                                corrected_hr_df, corrected_rr_df,
                                fs,
                                high_pass,
                                low_pass,
                                delta,
                                peak_rise)

    if outpath is not None:
        filepath = opj(outpath, "ecg_qc.html")
        report.save_as_html(filepath)
        print(f"QC report for ECG signal saved in: {filepath}")

    # Store filtered data and peaks in a dictionary for output
    output_dict = {'filtered_signal': signal_filt,
                   'peaks': corrected_peaks
                   }

    return report, output_dict


def plot_filtered_signal(ax, signal, signal_filt):
    # plots comparison between unfiltered and filtered signal (one panel)
    ax.plot(signal, label="unfiltered signal")
    ax.plot(signal_filt, label="filtered signal")
    # ax.set_xlim([5000, 7000])
    ax.legend()
    return ax


def plot_power_spectrum_ecg(ax, signal, signal_filt, fs):
    # plots power spectrum of unfiltered, filtered signal (one panel)
    f, Pxx = welch(signal, fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="unfiltered signal")
    f, Pxx = welch(signal_filt, fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="filtered signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectrum")
    ax.set_xlim([0, 20])
    ax.legend()

    return ax


def plot_rr_hist(ax, diff_peaks):

    ax.hist(diff_peaks, bins=50, density=True)
    ax.set_xlabel("RR interval (ms)")
    ax.set_ylabel("Probability density")

    return ax


def plot_inst_hr(ax, fs, diff_peaks):

    inst_hr = (fs/diff_peaks)*60
    ax.plot(inst_hr)
    ax.set_ylim(30, 120)
    # ax.set_xlabel("RR interval number")
    ax.set_ylabel("BPM")

    return ax, inst_hr


def plot_filtered_data(signal, signal_filt, fs, peak_rise, delta):

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12))

    # Limits for signal plots of 5 secs
    x_i = 0
    x_f = fs*5
    if signal.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal.shape[0]
    ax1 = axs[0, 0]
    ax1 = plot_filtered_signal(ax1, signal, signal_filt)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f])

    ax2 = axs[0, 1]
    ax2 = plot_power_spectrum_ecg(ax2, signal, signal_filt, fs)
    ax2.set_title("B", size=15)

    # Compute peaks
    peaks = compute_max_events(signal_filt,
                               peak_rise=peak_rise,
                               delta=delta)
    diff_peaks = abs(np.diff(peaks))
    # Heart rate using the difference time between peaks
    heart_rate = np.mean(fs/diff_peaks)*60

    ax3 = axs[1, 0]
    ax3 = plot_peaks(ax3, signal_filt, peaks)
    ax3.set_title("C", size=15)
    ax3.set_xlim([x_i, x_f])

    # Compute signal around peaks
    ax4 = axs[1, 1]
    ax4 = plot_average_signal(ax4, peaks, delta, signal_filt)
    ax4.set_title("D: Heart rate = %.2f bpm" % heart_rate, size=15)

    # Compute mean, median, stdev, snr of RR interval
    mean_RR, median_RR, stdev_RR, snr_RR = compute_stats(diff_peaks)

    # Compute peaks and plot histogram of RR interval
    ax5 = axs[2, 0]
    ax5 = plot_rr_hist(ax5, diff_peaks)
    ax5.set_title("E: RR mean = %.2f, "
                  "median = %.2f, "
                  "stdev = %.2f" % (mean_RR, median_RR, stdev_RR),
                  size=13)

    # Compute and plot instantaneous HR
    ax6 = axs[2, 1]
    ax6, inst_hr = plot_inst_hr(ax6, fs, diff_peaks)
    ax6.set_title("F: Instantaneous heart rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig, peaks, diff_peaks, heart_rate, mean_RR, median_RR, stdev_RR, snr_RR


def plot_corrected_ecg(signal_filt, peaks, diff_peaks, delta, fs):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

    # As before, limits for signal plots of 5 secs

    x_i = 0
    x_f = fs*5
    if signal_filt.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal_filt.shape[0]

    ax1 = axs[0, 0]
    ax1 = plot_peaks(ax1, signal_filt, peaks)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f])

    # Heart rate using the difference time between peaks
    heart_rate = np.mean(fs/diff_peaks)*60

    # Compute signal around peaks
    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, peaks, delta, signal_filt)
    ax2.set_title("B: Corrected HR = %.2f bpm" % heart_rate, size=15)

    # Compute mean, median, stdev, snr of RR interval
    mean_RR, median_RR, stdev_RR, snr_RR = compute_stats(diff_peaks)

    # Compute peaks and plot histogram of RR interval
    ax3 = axs[1, 0]
    ax3 = plot_rr_hist(ax3, diff_peaks)
    ax3.set_title("C: RR mean = %.2f, "
                  "median = %.2f,stdev = %.2f" % (mean_RR,
                                                  median_RR,
                                                  stdev_RR),
                  size=15)

    # Compute and plot instantaneous HR
    ax4 = axs[1, 1]
    ax4, inst_hr = plot_inst_hr(ax4, fs, diff_peaks)
    ax4.set_title("D: Corrected Instantaneous heart rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig, heart_rate, mean_RR, median_RR, stdev_RR, snr_RR, inst_hr


def plot_comparison_ecg(signal_filt, peaks, diff_peaks, heart_rate,
                        mean_RR, median_RR, stddev_RR,
                        corrected_peaks, corrected_diff_peaks2,
                        corrected_heart_rate, c_mean_RR, c_median_RR,
                        c_stdev_RR, delta, fs):

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12))

    ax1 = axs[0, 0]
    ax1 = plot_average_signal(ax1, peaks, delta, signal_filt)
    ax1.set_title("A: HR = %.2f bpm" % heart_rate, size=15)

    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, corrected_peaks, delta, signal_filt)
    ax2.set_title("B: Corrected HR = %.2f bpm" % corrected_heart_rate, size=15)

    # Plot histogram of RR interval
    ax3 = axs[1, 0]
    ax3 = plot_rr_hist(ax3, diff_peaks)
    ax3.set_title("C: RR mean = %.2f, "
                  "median = %.2f, "
                  "stdev = %.2f" % (mean_RR, median_RR, stddev_RR),
                  size=15)

    ax4 = axs[1, 1]
    ax4 = plot_rr_hist(ax4, corrected_diff_peaks2)
    ax4.set_title("D: RR mean = %.2f, "
                  "median = %.2f, "
                  "stdev = %.2f" % (c_mean_RR,
                                    c_median_RR,
                                    c_stdev_RR),
                  size=15)

    # Plot instantaneous HR
    ax5 = axs[2, 0]
    ax5, inst_hr = plot_inst_hr(ax5, fs, diff_peaks)
    ax5.set_title("E: Instantaneous heart rate", size=15)

    ax6 = axs[2, 1]
    ax6, c_inst_hr = plot_inst_hr(ax6, fs, corrected_diff_peaks2)
    ax6.set_title("F: Corrected instantaneous heart rate", size=15)

    fig.tight_layout()
    plt.close()

    return fig


def _generate_ecg_html(fig1, fig2, fig3, hr_df, rr_df,
                       max_indices, min_indices,
                       corrected_hr_df, corrected_rr_df,
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
                                           'report_body_ecg.html')

    with open(html_head_template_path) as html_head_file_obj:
        html_head_template_text = html_head_file_obj.read()
    report_head_template = string.Template(html_head_template_text)

    with open(html_body_template_path) as html_body_file_obj:
        html_body_template_text = html_body_file_obj.read()
    report_body_template = string.Template(html_body_template_text)

    page_title = 'niphlem'
    page_heading = ('niphlem: ECG signal processing and '
                    'peak detection QC report')

    fig1_html = _plot_to_svg(fig1)
    fig2_html = _plot_to_svg(fig2)
    fig3_html = _plot_to_svg(fig3)

    hr_html = _dataframe_to_html(hr_df,
                                 precision=2,
                                 header=True,
                                 sparsify=False
                                 )

    rr_html = _dataframe_to_html(rr_df,
                                 precision=2,
                                 header=True,
                                 sparsify=False,
                                 )

    corrected_hr_html = _dataframe_to_html(corrected_hr_df,
                                           precision=2,
                                           header=True,
                                           sparsify=False,
                                           )

    corrected_rr_html = _dataframe_to_html(corrected_rr_df,
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
                          'hr_html': hr_html,
                          'rr_html': rr_html,
                          'max_indices': max_indices,
                          'min_indices': min_indices,
                          'fig2_html': fig2_html,
                          'corrected_hr_html': corrected_hr_html,
                          'corrected_rr_html': corrected_rr_html,
                          'fig3_html': fig3_html
                          }

    report_text_body = report_body_template.safe_substitute(**report_values_body)
    report_text = HTMLReport(body=report_text_body,
                             head_tpl=report_head_template,
                             head_values=report_values_head)

    return report_text