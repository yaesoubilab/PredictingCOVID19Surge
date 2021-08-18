import matplotlib.pyplot as plt
import numpy as np

from SimPy.InOutFunctions import read_csv_rows
from definitions import ROOT_DIR

X_LABEL_COLORS = ['blue', 'green']


def add_to_ax(ax, title, x_labels, ys, errs, colors, show_y_label):

    x_pos = np.arange(len(x_labels))
    ax.scatter(x_pos, ys, c=colors)
    for pos, y, err, color in zip(x_pos, ys, errs, colors):
        ax.errorbar(pos, y, yerr=err, fmt="o", c=color)

    if show_y_label:
        ax.set_ylabel('$R^{2}$ Score')
    ax.set_xlabel('Predictive Models')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim((0, 1))
    ax.set_xlim((-0.5, len(x_labels) - 0.5))
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.5)


def plot_performance():
    # read data
    data = read_csv_rows(file_name=ROOT_DIR+'/outputs/prediction_summary/summary.csv',
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
    fig, axes = plt.subplots(1, len(dict_of_figs), figsize=(7.5, 4))

    i = 0
    for key, value in dict_of_figs.items():
        add_to_ax(ax=axes[i],
                  title=key,
                  x_labels=[v[0] for v in value],
                  ys=[v[1] for v in value],
                  errs=[v[2] for v in value],
                  colors=X_LABEL_COLORS,
                  show_y_label=True if i == 0 else False)

        i += 1

    # Save the figure and show
    plt.tight_layout()
    plt.savefig(ROOT_DIR+'/outputs/figures/performance.png')
    plt.show()


if __name__ == '__main__':
    plot_performance()
