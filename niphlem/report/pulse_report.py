"""
This file contains the code to generate the pulse-ox report.

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
                             generate_rate_df, generate_interval_df)
from .html_report import HTMLReport


def make_pulseox_report(pulse_signal,
                        *,
                        fs,
                        delta_low,
                        delta_high,
                        peak_rise_low=0.5,
                        peak_rise_high=0.75,
                        resp_high_pass=0.1,
                        resp_low_pass=0.5,
                        hr_high_pass=0.6,
                        hr_low_pass=5,
                        outpath=None,
                        ):
    """
    Generate QC report for pulse-ox data.
    Parameters
    ----------
    pulse_signal : array-like of shape (n_physio_samples, ),
        or (n_physio_samples, n_channels).
        Pulse-oximetry belt signal.
    fs : float
        Sampling frequency of pulse-oximetry recording.
    delta_low, delta_high: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise_low, peak_rise_high: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak. The default is 0.5 and 0.75 respectively.
    resp_high_pass, hr_high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 0.1 and 0.6 respectively.
    resp_low_pass, hr_low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is 0.5 and 5 respectively.
    outpath : string, optional
        If provided, Path where report the HTML report,
        averaged filtered signal and corrected peaks will be saved.
        The default is None.
    Returns
    -------
    report : html file
        HTML report.
    output_dict : dict
        Dictionary with the (high and low) filtered signals and
        (corrected) peak locations.
    """

    signal = pulse_signal.copy()

    signal = validate_signal(signal)

    outpath = validate_outpath(outpath)

    # demean and lowest frequencies filtering
    signal_filt_low = np.apply_along_axis(_transform_filter,
                                          axis=0,
                                          arr=signal,
                                          high_pass=resp_high_pass,
                                          low_pass=resp_low_pass,
                                          sampling_rate=fs
                                          )
    # Compute average signal across channels of filter data
    signal_filt_low = np.mean(signal_filt_low, axis=1)

    # demean and highest frequencies filtering
    signal_filt_high = np.apply_along_axis(_transform_filter,
                                           axis=0,
                                           arr=signal,
                                           high_pass=hr_high_pass,
                                           low_pass=hr_low_pass,
                                           sampling_rate=fs
                                           )
    # Compute average signal across channels of filter data
    signal_filt_high = np.mean(signal_filt_high, axis=1)

    # Average raw signal
    signal = np.mean(signal, axis=1)

    if outpath is not None:
        filepath_low = opj(outpath, "transformed_signal_low_puls.txt")
        np.savetxt(filepath_low, signal_filt_low)
        print("Transformed low frequency pulse-ox "
              f"signal saved in: {filepath_low}"
              )

        filepath_high = opj(outpath, "transformed_signal_high_puls.txt")
        np.savetxt(filepath_high, signal_filt_high)
        print("Transformed high frequency pulse-ox "
              f"signal saved in: {filepath_high}"
              )

    fig0 = plot_combined_transformed_data(signal,
                                          signal_filt_low,
                                          signal_filt_high,
                                          fs)

    # respiration signal
    labels_resp = ["Respiration", "IPI", "IPI", "respiration"]
    fig1, resp_peaks, resp_diff_peaks, resp_rate, \
        mean_ipi, median_ipi, stdev_ipi, snr_ipi, = plot_transformed_pulse(
            signal_filt_low,
            fs,
            peak_rise_low,
            delta_low,
            labels_resp
            )

    rr_df = generate_rate_df(fs, resp_diff_peaks, resp_rate)
    ipi_df = generate_interval_df(mean_ipi, median_ipi, stdev_ipi, snr_ipi)

    corrected_resp_peaks, \
        resp_max_indices, resp_min_indices = correct_anomalies(resp_peaks)

    # Compute differences between corrected peaks
    corrected_resp_peak_diffs = abs(np.diff(corrected_resp_peaks))

    if outpath is not None:
        filepath_low = opj(outpath, "peaks_low_puls.txt")
        np.savetxt(filepath_low, corrected_resp_peaks)
        print("Peaks of the low frequency pulse-ox "
              f"signal saved in: {filepath_low}"
              )

    fig2, c_resp_rate, c_mean_ipi, c_median_ipi, \
        c_stdev_ipi, c_snr_ipi, c_inst_resp = plot_corrected_pulse(
            signal_filt_low,
            corrected_resp_peaks,
            corrected_resp_peak_diffs,
            delta_low, fs, labels_resp
            )

    corrected_rr_df = generate_rate_df(fs,
                                       corrected_resp_peak_diffs,
                                       c_resp_rate
                                       )
    corrected_ipi_df = generate_interval_df(c_mean_ipi,
                                            c_median_ipi,
                                            c_stdev_ipi,
                                            c_snr_ipi
                                            )

    fig3 = plot_comparison_pulse(signal_filt_low, resp_peaks,
                                 resp_diff_peaks, resp_rate,
                                 mean_ipi, median_ipi, stdev_ipi,
                                 corrected_resp_peaks,
                                 corrected_resp_peak_diffs,
                                 c_resp_rate, c_mean_ipi,
                                 c_median_ipi,
                                 c_stdev_ipi,
                                 delta_low, fs,
                                 labels_resp)

    # cardiac signal
    labels_hr = ["Heart", "RR", "RR interval", "heart"]
    fig4, hr_peaks, hr_diff_peaks, heart_rate, \
        mean_rr, median_rr, stdev_rr, snr_rr = plot_transformed_pulse(
            signal_filt_high,
            fs,
            peak_rise_high,
            delta_high,
            labels_hr
            )

    hr_df = generate_rate_df(fs, hr_diff_peaks, heart_rate)
    rri_df = generate_interval_df(mean_rr, median_rr, stdev_rr, snr_rr)

    corrected_hr_peaks, \
        hr_max_indices, hr_min_indices = correct_anomalies(hr_peaks)
    # Compute differences between corrected peaks
    corrected_hr_peak_diffs = abs(np.diff(corrected_hr_peaks))

    if outpath is not None:
        filepath_high = opj(outpath, "peaks_high_puls.txt")
        np.savetxt(filepath_high, corrected_hr_peaks)
        print("Peaks of the high frequency pulse-ox "
              f"signal saved in: {filepath_high}"
              )

    fig5, c_heart_rate, c_mean_rr, c_median_rr, \
        c_stdev_rr, c_snr_rr, c_inst_hr = plot_corrected_pulse(
            signal_filt_high,
            corrected_hr_peaks,
            corrected_hr_peak_diffs,
            delta_high, fs, labels_hr
            )

    corrected_hr_df = generate_rate_df(fs,
                                       corrected_hr_peak_diffs,
                                       c_heart_rate)
    corrected_rri_df = generate_interval_df(c_mean_rr,
                                            c_median_rr,
                                            c_stdev_rr,
                                            c_snr_rr)

    fig6 = plot_comparison_pulse(signal_filt_high, hr_peaks,
                                 hr_diff_peaks, heart_rate,
                                 mean_rr, median_rr, stdev_rr,
                                 corrected_hr_peaks, corrected_hr_peak_diffs,
                                 c_heart_rate, c_mean_rr,
                                 c_median_rr, c_stdev_rr,
                                 delta_high, fs, labels_hr)

    # generate html report
    report = _generate_pulse_html(fig0, fig1, fig2, fig3, fig4, fig5, fig6,
                                  rr_df, hr_df, ipi_df, rri_df,
                                  resp_max_indices, resp_min_indices,
                                  hr_max_indices, hr_min_indices,
                                  corrected_rr_df, corrected_hr_df,
                                  corrected_ipi_df, corrected_rri_df,
                                  fs, resp_high_pass, resp_low_pass,
                                  hr_high_pass, hr_low_pass,
                                  delta_low, delta_high,
                                  peak_rise_low, peak_rise_high)

    if outpath is not None:
        filepath = opj(outpath, "pulseox_qc.html")
        report.save_as_html(filepath)
        print(f"QC report for pulse-ox signal saved in: {filepath}")

    # Store filtered data and peaks in a dictionary for output
    output_dict = {'low_filtered_signal': signal_filt_low,
                   'high_filtered_signal': signal_filt_high,
                   'peaks_low': corrected_resp_peaks,
                   'peaks_high': corrected_hr_peaks,
                   }

    return report, output_dict


def plot_transformed_signal_pulse(ax,
                                  signal,
                                  signal_filt_low,
                                  signal_filt_high):
    # plots comparison between raw and transformed signal (one panel)
    ax.plot(signal, label="raw signal")
    ax.plot(signal_filt_low, label="low transformed signal")
    ax.plot(signal_filt_high, label="high transformed signal")
    ax.legend()
    return ax


def plot_power_spectrum_pulse(ax,
                              signal,
                              signal_filt_low,
                              signal_filt_high,
                              fs):
    # plots power spectrum of raw, transformed signal (one panel)
    f, Pxx = welch(signal.flatten(), fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="raw signal")
    f, Pxx = welch(signal_filt_low, fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="low transformed signal")
    f, Pxx = welch(signal_filt_high, fs=fs, nperseg=2048, scaling="spectrum")
    ax.semilogy(f, Pxx, label="high transformed signal")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectrum")
    ax.set_xlim([0, 20])
    ax.legend()

    return ax


def plot_hist(ax, diff_peaks):

    ax.hist(diff_peaks, bins=50, density=True)
    ax.set_ylabel("Probability density")

    return ax


def plot_inst_signal(ax, fs, diff_peaks):

    inst_resp = (fs/diff_peaks)*60
    ax.plot(inst_resp)

    return ax, inst_resp


def plot_combined_transformed_data(signal,
                                   signal_filt_low,
                                   signal_filt_high,
                                   fs):

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))

    # Plot signal
    x_i = 0
    x_f = 10000
    if signal.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal.shape[0]
    ax1 = axs[0]
    ax1 = plot_transformed_signal_pulse(ax1,
                                        signal,
                                        signal_filt_low,
                                        signal_filt_high)
    ax1.set_xlim([x_i, x_f])

    # Show how frequencies are filtered
    ax2 = axs[1]
    ax2 = plot_power_spectrum_pulse(ax2,
                                    signal,
                                    signal_filt_low,
                                    signal_filt_high,
                                    fs)

    fig.tight_layout()
    plt.close()

    return fig


def plot_transformed_pulse(signal_filt, fs, peak_rise, delta, labels):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

    x_i = 0
    x_f = 10000
    if signal_filt.shape[0] < x_f:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f = signal_filt.shape[0]

    # Compute peaks
    # times = np.arange(signal.shape[0])*1/400
    peaks = compute_max_events(signal_filt,
                               peak_rise=peak_rise,
                               delta=delta)
    diff_peaks = abs(np.diff(peaks))
    # Signal rate using the difference time between peaks
    signal_rate = np.mean(fs/diff_peaks)*60
    ax1 = axs[0, 0]
    ax1 = plot_peaks(ax1, signal_filt, peaks)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f])

    # Compute signal around peaks
    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, peaks, delta, signal_filt)
    ax2.set_title("B: %s rate = %.2f bpm" % (labels[0], signal_rate), size=15)

    # Compute mean, median, stdev, snr of IPI
    mean_ipi, median_ipi, stdev_ipi, snr_ipi = compute_stats(diff_peaks)

    # Compute peaks and plot histogram of IPI
    ax3 = axs[1, 0]
    ax3 = plot_hist(ax3, diff_peaks)
    ax3.set_title("C: %s mean = %.2f, "
                  "median = %.2f, "
                  "stdev = %.2f" % (labels[1], mean_ipi,
                                    median_ipi, stdev_ipi),
                  size=13)
    ax3.set_xlabel("%s (ms)" % labels[2])

    # Compute and plot instantaneous signal rate
    ax4 = axs[1, 1]
    ax4, inst_signal = plot_inst_signal(ax4, fs, diff_peaks)
    ax4.set_title("D: Instantaneous %s rate" % labels[3], size=15)
    ax4.set_ylabel("BPM")

    fig.tight_layout()
    plt.close()

    return fig, peaks, diff_peaks, signal_rate, mean_ipi, median_ipi, stdev_ipi, snr_ipi


def plot_corrected_pulse(signal_filt,
                         corrected_peaks,
                         corrected_peak_diffs,
                         delta,
                         fs,
                         labels):

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))

    x_i = 0
    x_f1 = 10000
    if signal_filt.shape[0] < x_f1:
        # In the unlikely case where mean signal duration is less than 5 secs
        x_f1 = signal_filt.shape[0]

    ax1 = axs[0, 0]
    ax1 = plot_peaks(ax1, signal_filt, corrected_peaks)
    ax1.set_title("A", size=15)
    ax1.set_xlim([x_i, x_f1])

    # Signal rate using the difference time between peaks
    corrected_signal_rate = np.mean(fs/corrected_peak_diffs)*60

    # Compute signal around peaks
    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, corrected_peaks, delta, signal_filt)
    ax2.set_title("B: Corrected %s rate = %.2f bpm" % (labels[3],
                                                       corrected_signal_rate),
                  size=15)

    # Compute mean, median, stdev, snr of IPI
    mean_ipi, median_ipi, \
        stdev_ipi, snr_ipi = compute_stats(corrected_peak_diffs)

    # Compute peaks and plot histogram of IPI
    ax3 = axs[1, 0]
    ax3 = plot_hist(ax3, corrected_peak_diffs)
    ax3.set_title("C: %s mean = %.2f, "
                  "median = %.2f, stdev = %.2f" % (labels[1],
                                                   mean_ipi,
                                                   median_ipi,
                                                   stdev_ipi),
                  size=15)
    ax3.set_xlabel("%s (ms)" % labels[2])

    # Compute and plot instantaneous signal rate
    ax4 = axs[1, 1]
    ax4, inst_signal = plot_inst_signal(ax4, fs, corrected_peak_diffs)
    ax4.set_title("D: Corrected instantaneous %s rate" % labels[3], size=15)
    ax4.set_ylabel("BPM")

    fig.tight_layout()
    plt.close()

    return fig, corrected_signal_rate, mean_ipi, median_ipi, stdev_ipi, snr_ipi, inst_signal


def plot_comparison_pulse(signal_filt, peaks, diff_peaks, signal_rate,
                          mean_ipi, median_ipi, stdev_ipi,
                          corrected_peaks, corrected_peak_diffs,
                          c_signal_rate, c_mean_ipi, c_median_ipi, c_stdev_ipi,
                          delta, fs, labels):

    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 12))

    ax1 = axs[0, 0]
    ax1 = plot_average_signal(ax1, peaks, delta, signal_filt)
    ax1.set_title("A: %s rate = %.2f bpm" % (labels[0], signal_rate), size=15)

    ax2 = axs[0, 1]
    ax2 = plot_average_signal(ax2, corrected_peaks, delta, signal_filt)
    ax2.set_title("B: Corrected %s rate = %.2f bpm" % (labels[3],
                                                       c_signal_rate),
                  size=15)

    # Plot histogram of IPI
    ax3 = axs[1, 0]
    ax3 = plot_hist(ax3, diff_peaks)
    ax3.set_title("C: %s mean = %.2f, "
                  "med = %.2f, "
                  "stdev = %.2f" % (labels[1],
                                    mean_ipi,
                                    median_ipi,
                                    stdev_ipi),
                  size=15)
    ax3.set_xlabel("%s (ms)" % labels[2])

    ax4 = axs[1, 1]
    ax4 = plot_hist(ax4, corrected_peak_diffs)
    ax4.set_title("D: %s mean = %.2f, "
                  "med = %.2f, "
                  "stdev = %.2f" % (labels[1],
                                    c_mean_ipi,
                                    c_median_ipi,
                                    c_stdev_ipi),
                  size=15)
    ax3.set_xlabel("%s (ms)" % labels[2])

    # Plot instantaneous signal rate
    ax5 = axs[2, 0]
    ax5, inst_signal = plot_inst_signal(ax5, fs, diff_peaks)
    ax5.set_title("E: Instantaneous %s rate" % labels[3], size=15)
    ax5.set_ylabel("BPM")

    ax6 = axs[2, 1]
    ax6, c_inst_signal = plot_inst_signal(ax6, fs, corrected_peak_diffs)
    ax6.set_title("F: Corrected instantaneous %s rate" % labels[3], size=15)
    ax6.set_ylabel("BPM")

    fig.tight_layout()
    plt.close()

    return fig


def _generate_pulse_html(fig0, fig1, fig2, fig3, fig4, fig5, fig6,
                         rr_df, hr_df, ipi_df, rri_df,
                         resp_max_indices, resp_min_indices,
                         hr_max_indices, hr_min_indices,
                         corrected_rr_df, corrected_hr_df,
                         corrected_ipi_df, corrected_rri_df,
                         fs, resp_high_pass, resp_low_pass,
                         hr_high_pass, hr_low_pass,
                         delta_low, delta_high,
                         peak_rise_low, peak_rise_high
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
                                           'report_body_pulse.html')

    with open(html_head_template_path) as html_head_file_obj:
        html_head_template_text = html_head_file_obj.read()
    report_head_template = string.Template(html_head_template_text)

    with open(html_body_template_path) as html_body_file_obj:
        html_body_template_text = html_body_file_obj.read()
    report_body_template = string.Template(html_body_template_text)

    page_title = 'niphlem'
    page_heading = ('niphlem: pulse-oximetry signal processing and '
                    'peak detection QC report')

    fig0_html = _plot_to_svg(fig0)
    fig1_html = _plot_to_svg(fig1)
    fig2_html = _plot_to_svg(fig2)
    fig3_html = _plot_to_svg(fig3)
    fig4_html = _plot_to_svg(fig4)
    fig5_html = _plot_to_svg(fig5)
    fig6_html = _plot_to_svg(fig6)

    rr_html = _dataframe_to_html(rr_df,
                                 precision=2,
                                 header=True,
                                 sparsify=False
                                 )

    hr_html = _dataframe_to_html(hr_df,
                                 precision=2,
                                 header=True,
                                 sparsify=False
                                 )

    ipi_html = _dataframe_to_html(ipi_df,
                                  precision=2,
                                  header=True,
                                  sparsify=False,
                                  )

    rri_html = _dataframe_to_html(rri_df,
                                  precision=2,
                                  header=True,
                                  sparsify=False,
                                  )

    corrected_rr_html = _dataframe_to_html(corrected_rr_df,
                                           precision=2,
                                           header=True,
                                           sparsify=False,
                                           )

    corrected_hr_html = _dataframe_to_html(corrected_hr_df,
                                           precision=2,
                                           header=True,
                                           sparsify=False,
                                           )

    corrected_ipi_html = _dataframe_to_html(corrected_ipi_df,
                                            precision=2,
                                            header=True,
                                            sparsify=False,
                                            )

    corrected_rri_html = _dataframe_to_html(corrected_rri_df,
                                            precision=2,
                                            header=True,
                                            sparsify=False,
                                            )

    report_values_head = {'page_title': escape(page_title)}
    report_values_body = {'page_heading': page_heading,
                          'fig0_html': fig0_html,
                          'fs': fs,
                          'resp_low_cut': resp_high_pass,
                          'resp_high_cut': resp_low_pass,
                          'delta_low': delta_low,
                          'peak_rise_low': peak_rise_low,
                          'fig1_html': fig1_html,
                          'rr_html': rr_html,
                          'ipi_html': ipi_html,
                          'resp_max_indices': resp_max_indices,
                          'resp_min_indices': resp_min_indices,
                          'fig2_html': fig2_html,
                          'corrected_rr_html': corrected_rr_html,
                          'corrected_ipi_html': corrected_ipi_html,
                          'fig3_html': fig3_html,
                          'hr_low_cut': hr_high_pass,
                          'hr_high_cut': hr_low_pass,
                          'delta_high': delta_high,
                          'peak_rise_high': peak_rise_high,
                          'fig4_html': fig4_html,
                          'hr_html': hr_html,
                          'rri_html': rri_html,
                          'hr_max_indices': hr_max_indices,
                          'hr_min_indices': hr_min_indices,
                          'fig5_html': fig5_html,
                          'corrected_hr_html': corrected_hr_html,
                          'corrected_rri_html': corrected_rri_html,
                          'fig6_html': fig6_html
                          }

    report_text_body = report_body_template.safe_substitute(**report_values_body)
    report_text = HTMLReport(body=report_text_body,
                             head_tpl=report_head_template,
                             head_values=report_values_head)

    return report_text
