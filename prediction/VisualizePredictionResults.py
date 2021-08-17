import matplotlib.pyplot as plt
import numpy as np


def add_to_ax(ax, week, x_labels, ys, errs, colors):

    x_pos = np.arange(len(x_labels))
    ax.scatter(x_pos, ys, c=colors)
    for pos, y, err, color in zip(x_pos, ys, errs, colors):
        ax.errorbar(pos, y, yerr=err, fmt="o", c=color)

    ax.set_ylabel('$R^{2}$ Score')
    ax.set_xlabel('Predictive Models')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim((0, 5))
    ax.set_xlim((-0.5, len(x_labels) - 0.5))
    ax.set_title('{} weeks until the peak'.format(week))
    ax.yaxis.grid(True)


fig, axes = plt.subplots(1, 3, figsize=(7, 4))
add_to_ax(ax=axes[0], week=4,
          x_labels=['Full', 'A', 'B'],
          ys=[1, 2, 3],
          errs=[0.4, 0.3, 0.2],
          colors=['blue', 'green', 'red'])


# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()
