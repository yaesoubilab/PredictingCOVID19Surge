import string

import matplotlib.pyplot as plt
import numpy as np

from SimPy.InOutFunctions import read_csv_rows
from SimPy.Plots.FigSupport import output_figure
from SimPy.Statistics import SummaryStat
from definitions import ROOT_DIR, get_dataset_labels, get_short_outcome

X_LABEL_COLORS = ['black', 'purple', 'magenta', 'blue', 'cyan', 'green', 'orange', 'red', 'brown']
Y_LABELS = ['Predicting the size of\nhospitalization peak\n($R^{2}$ Score)',
            'Predicting if hospitalization\ncapacity will be exceeded\n(ROC AUC)']
FIG_SIZE = (9, 6)
N_OF_PRED_TIMES = 3 # 12, 8, 4 weeks to peak


def add_to_ax(ax, title, panel_label, x_labels, ys, errs, colors,
              y_range, y_label, show_y_values, show_x_label):

    x_pos = np.arange(len(x_labels))
    ax.scatter(x_pos, ys, c=colors)
    for pos, y, err, color in zip(x_pos, ys, errs, colors):
        ax.errorbar(pos, y, yerr=[[err[0]], [err[1]]], fmt="o", c=color)

    # labels
    ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
            size=10, weight='bold')

    ax.set_ylabel(y_label)
    if not show_y_values:
        ax.set_yticklabels([])
    if show_x_label:
        ax.set_xlabel('Predictive Models')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(y_range)
    ax.set_xlim((-0.5, len(x_labels) - 0.5))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.5)


def add_performance_for_outcome(axes, short_outcome, panel_labels, show_x_label, noise_coeff=None, bias_delay=None):

    # find the file name
    label = get_dataset_labels(
        week=None, survey_size=noise_coeff, bias_delay=bias_delay)

    # read data
    data = read_csv_rows(
        file_name=ROOT_DIR+'/outputs/prediction_summary/neu_net/predicting {}-summary{}.csv'.format(short_outcome, label),
        if_ignore_first_row=True, if_convert_float=True)

    dict_of_figs = {}
    for row in data:
        if int(row[0]) < 0:
            title = '{} Weeks until Peak'.format(-int(row[0]))
        else:
            title = '{} Weeks into Fall'.format(int(row[0]))

        # error in form [a, b]
        confidence_interval = SummaryStat.get_array_from_formatted_interval(interval=row[3])
        error = [row[2]-confidence_interval[0], confidence_interval[1]-row[2]]

        if title in dict_of_figs:
            # (model, mean, error)
            dict_of_figs[title].append([row[1], row[2], error])
        else:
            # (model, mean, error)
            dict_of_figs[title] = [[row[1], row[2], error]]

    i = 0
    for key, value in dict_of_figs.items():

        x_labels = []
        for v in value:
            if type(v[0]) == float:
                x_labels.append(str(int(v[0])))
            else:
                x_labels.append(v[0])

        # y-labels
        if i == 0:
            if short_outcome == 'size':
                y_label = Y_LABELS[0]
                y_range = (0, 1)
            elif short_outcome == 'prob':
                y_label = Y_LABELS[1]
                y_range = (0.5, 1)
            else:
                raise ValueError('Invalid short outcome.')
        else:
            y_label = ' '

        add_to_ax(ax=axes[i],
                  title=key,
                  panel_label=panel_labels[i],
                  x_labels=x_labels,
                  ys=[v[1] for v in value],
                  errs=[v[2] for v in value],
                  colors=X_LABEL_COLORS[:len(value)],
                  y_range=y_range,
                  y_label=y_label,
                  show_y_values=True if i == 0 else False,
                  show_x_label=show_x_label)

        i += 1


def plot_performance(noise_coeff=None, bias_delay=None, fig_size=None):

    fig_size = FIG_SIZE if fig_size is None else fig_size

    # make the figure
    fig, axes = plt.subplots(2, N_OF_PRED_TIMES, figsize=fig_size)

    i = 0
    for outcome in ('Maximum hospitalization rate', 'If hospitalization threshold passed'):

        # short outcome
        short_outcome = get_short_outcome(outcome=outcome)

        # panel labels
        panel_labels = [string.ascii_uppercase[N_OF_PRED_TIMES*i+j] + ')' for j in range(N_OF_PRED_TIMES)]

        add_performance_for_outcome(
            axes=axes[i], short_outcome=short_outcome, panel_labels=panel_labels,
            noise_coeff=noise_coeff, bias_delay=bias_delay,
            show_x_label=True if i == 1 else 0
        )

        i += 1

    # Save the figure and show
    label = get_dataset_labels(
        week=None, survey_size=noise_coeff, bias_delay=bias_delay)
    fig.tight_layout()
    output_figure(plt=fig,
                  filename=ROOT_DIR + '/outputs/figures/prediction/neu_net/performance{}.png'
                  .format(label))
    fig.show()


if __name__ == '__main__':

    plot_performance(noise_coeff=None, fig_size=FIG_SIZE)
    # plot_performance(noise_coeff=1, fig_size=FIG_SIZE)
    # plot_performance(noise_coeff=0.5, bias_delay=4, fig_size=FIG_SIZE)
