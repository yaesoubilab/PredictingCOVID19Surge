import matplotlib.pyplot as plt
import numpy as np

SIM_DURATION = 2.25
T0 = 0
X_RANGE = (-0.1, 2.35)
Y_RANGE = (0, 1)
SEPTEMBER_FIRST = 1.5

def f(t, b, t_mid, b_min, b_max):
    # sigmoid function f(t) = min + (max-min) * 1 / (1 + exp(-b * (t - t_mid))
    return b_min + (b_max-b_min) / (1 + np.exp(-b * (t - t_mid)))


def plot_sigmoid_functions(b, t_mid, b_min, b_max,
                           bs, t_mids, b_mins, b_maxs,
                           y_label):

    # ------------------
    ts = np.linspace(start=T0, stop=SIM_DURATION)
    fs = []
    legends = []
    titles = []

    # varying b_max
    if b_maxs is not None:
        titles.append('Varying ' + r'$b_{max}$')
        legs = []  # legends
        ys = []
        for v in b_maxs:
            legs.append(r'$b_{max}=$' + str(v))
            ys.append([f(t, b=b, t_mid=t_mid, b_min=b_min, b_max=v) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b_min
    if b_mins is not None:
        titles.append('Varying ' + r'$b_{min}$')
        legs = []  # legends
        ys = []
        for v in b_mins:
            legs.append(r'$b_{min}=$' + str(v))
            ys.append([f(t, b=b, t_mid=t_mid, b_min=v, b_max=b_max) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b
    if bs is not None:
        titles.append('Varying ' + r'$b$')
        legs = []  # legends
        ys = []
        for v in bs:
            legs.append(r'$b=$' + str(v))
            ys.append([f(t, b=v, t_mid=t_mid, b_min=b_min, b_max=b_max) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying t_mid
    if ts is not None:
        titles.append('Varying ' + r'$t_{mid}$')
        legs = []  # legends
        ys = []
        for v in t_mids:
            legs.append(r'$t_{mid}=$' + str(v))
            ys.append([f(t, b=b, t_mid=v, b_min=b_min, b_max=b_max) for t in ts])
        legends.append(legs)
        fs.append(ys)

    fig, axarr = plt.subplots(1, len(fs), sharey=True, figsize=(7.5, 3.2))
    for i, ax in enumerate(axarr):
        ax.set_title(titles[i])

        for j in range(3):
            ax.plot(ts, fs[i][j], label=legends[i][j])  # color='b', linestyle='-')

        ax.set_ylim(Y_RANGE)
        ax.set_xlim(X_RANGE)
        ax.axvline(x=SEPTEMBER_FIRST, c='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel('Simulation Year ' + r'$(t)$')
        ax.legend(fontsize='x-small') # loc=2

    axarr[0].set_ylabel(y_label)
    plt.tight_layout()
    plt.show()


# ---- settings ----

# probability of novel strain over time
plot_sigmoid_functions(b=7, t_mid=1.25, b_min=0, b_max=0.5,
                       bs=[5, 7, 9], t_mids=[1, 1.25, 1.5], b_mins=None, b_maxs=[0.4, 0.5, 0.6],
                       y_label=r'$\gamma(t)$')
#
# plot_sigmoid_functions(b=7, t0=1.25, b_min=0, b_max=0.5,
#                        bs=[5, 7, 9], t0s=[1, 1.25, 1.5], b_mins=[0, 0.1, 0.2], b_maxs=[0.4, 0.5, 0.6])

Y_RANGE = (0, 2)
# T0 = 1
plot_sigmoid_functions(b=-8, t_mid=1.75, b_min=0.2, b_max=1.5,
                       bs=[-10, -8, -6], t_mids=[1.5, 1.75, 2], b_mins=[0, 0.2, 0.4], b_maxs=[1.25, 1.5, 1.75],
                       y_label=r'$v(t)$')
