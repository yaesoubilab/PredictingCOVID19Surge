import matplotlib.pyplot as plt
import numpy as np

from SimPy.InOutFunctions import read_csv_rows
from definitions import ROOT_DIR, get_dataset_labels

X_LABEL_COLORS = ['black', 'purple', 'magenta', 'blue', 'green', 'orange', 'red', 'brown']


def add_to_ax(ax, title, x_labels, ys, errs, colors, show_y_label, show_y_values):

    x_pos = np.arange(len(x_labels))
    ax.scatter(x_pos, ys, c=colors)
    for pos, y, err, color in zip(x_pos, ys, errs, colors):
        ax.errorbar(pos, y, yerr=err, fmt="o", c=color)

    if show_y_label:
        ax.set_ylabel('$R^{2}$ Score')
    else:
        ax.set_ylabel(' ')
    if not show_y_values:
        ax.set_yticklabels([])
    ax.set_xlabel('Predictive Models')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim((0, 1))
    ax.set_xlim((-0.5, len(x_labels) - 0.5))
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.5)


def plot_performance(noise_coeff=None, bias_delay=None, fig_size=(7.5, 4)):

    # find the file name
    label = get_dataset_labels(
        week=None, noise_coeff=noise_coeff, bias_delay=bias_delay)

    # read data
    data = read_csv_rows(file_name=ROOT_DIR+'/outputs/prediction_summary/summary{}.csv'.format(label),
                         if_ignore_first_row=True, if_convert_float=True)

    dict_of_figs = {}
    for row in data:
        if int(row[0]) < 0:
            title = '{} Weeks until Peak'.format(-int(row[0]))
        else:
            title = '{} Weeks into Fall'.format(int(row[0]))

        if title in dict_of_figs:
            dict_of_figs[title].append([row[1], row[2], row[3]])
        else:
            # (model, mean, error)
            dict_of_figs[title] = [[row[1], row[2], row[3]]]

    # make the figure
    fig, axes = plt.subplots(1, len(dict_of_figs), figsize=fig_size)

    i = 0
    for key, value in dict_of_figs.items():

        x_labels = []
        for v in value:
            if type(v[0]) == float:
                x_labels.append(str(int(v[0])))
            else:
                x_labels.append(v[0])

        add_to_ax(ax=axes[i],
                  title=key,
                  x_labels=x_labels,
                  ys=[v[1] for v in value],
                  errs=[v[2] for v in value],
                  colors=X_LABEL_COLORS[:len(value)],
                  show_y_label=True if i == 0 else False,
                  show_y_values=True if i == 0 else False)

        i += 1

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(ROOT_DIR+'/outputs/figures/performance{}.png'.format(label))
    plt.show()


if __name__ == '__main__':

    # noise could be None, 1, or 2
    plot_performance(noise_coeff=None, fig_size=(11, 3.6))
    plot_performance(noise_coeff=1, fig_size=(11, 3.6))
    plot_performance(noise_coeff=0.5, bias_delay=4, fig_size=(11, 3.6))
