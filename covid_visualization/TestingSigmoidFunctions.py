import matplotlib.pyplot as plt
import numpy as np

from SimPy.Parameters import Constant, TimeDependentSigmoid

SIM_DURATION = 2.25
T0 = 0
X_RANGE = (-0.1, 2.35)
Y_RANGE = (0, 1)
SEPTEMBER_FIRST = 1.5


def f(t, b, f_min, f_max, t_mid, t_min=0):
    # sigmoid function f(t) = min + (max-min) * 1 / (1 + exp(-b * (t - t_mid)) if t >= t_min
    # (0 otherwise)
    par = TimeDependentSigmoid(
        par_b=Constant(b),
        par_t_min=Constant(t_min),
        par_t_middle=Constant(t_mid),
        par_min=Constant(f_min),
        par_max=Constant(f_max)
    )
    val = par.sample(time=t)
    return val if val > 0 else None
    #
    # if t >= t_min:
    #     return b_min + (b_max-b_min) / (1 + np.exp(-b * (t - t_mid-t_min)))
    # else:
    #     return None


def plot_sigmoid_functions(b, f_min, f_max, t_mid, t_min,
                           bs, f_mins, f_maxs, t_mids, t_mins,
                           y_label, fig_size=(7.5, 3.2)):

    # ------------------
    ts = np.linspace(start=T0, stop=SIM_DURATION)
    fs = []
    legends = []
    titles = []

    # varying b
    if bs is not None:
        titles.append('Varying ' + r'$b$')
        legs = []  # legends
        ys = []
        for v in bs:
            legs.append(r'$b=$' + str(v))
            ys.append([f(t, b=v, t_mid=t_mid, f_min=f_min, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b_min
    if f_mins is not None:
        titles.append('Varying ' + r'$b_{min}$')
        legs = []  # legends
        ys = []
        for v in f_mins:
            legs.append(r'$b_{min}=$' + str(v))
            ys.append([f(t, b=b, t_mid=t_mid, f_min=v, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b_max
    if f_maxs is not None:
        titles.append('Varying ' + r'$b_{max}$')
        legs = []  # legends
        ys = []
        for v in f_maxs:
            legs.append(r'$b_{max}=$' + str(v))
            ys.append([f(t, b=b, t_mid=t_mid, f_min=f_min, f_max=v, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying t_mid
    if t_mids is not None:
        titles.append('Varying ' + r'$t_{mid}$')
        legs = []  # legends
        ys = []
        for v in t_mids:
            legs.append(r'$t_{mid}=$' + str(v))
            ys.append([f(t, b=b, t_mid=v, f_min=f_min, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying t_mins
    if t_mins is not None:
        titles.append('Varying ' + r'$t_{min}$')
        legs = []  # legends
        ys = []
        for v in t_mins:
            legs.append(r'$t_{min}=$' + str(v))
            ys.append([f(t, b=b, t_mid=t_mid, f_min=f_min, f_max=f_max, t_min=v) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # plot
    fig, axarr = plt.subplots(1, len(fs), sharey=True, figsize=fig_size)
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
plot_sigmoid_functions(b=7, bs=[5, 7, 9],
                       f_min=0, f_mins=None,
                       f_max=0.5, f_maxs=[0.4, 0.5, 0.6],
                       t_mid=1.75, t_mids=[1.5, 1.75, 2.0],
                       t_min=0, t_mins=None,
                       y_label=r'$\gamma(t)$', fig_size=(5.5, 3))
#
# plot_sigmoid_functions(b=7, t0=1.25, b_min=0, b_max=0.5,
#                        bs=[5, 7, 9], t0s=[1, 1.25, 1.5], b_mins=[0, 0.1, 0.2], b_maxs=[0.4, 0.5, 0.6])

Y_RANGE = (0, 2)
# rate of vaccination
plot_sigmoid_functions(b=-8, bs=[-10, -8, -6],
                       f_min=0.2, f_mins=[0, 0.2, 0.4],
                       f_max=1.5, f_maxs=[1.25, 1.5, 1.75],
                       t_mid=0.6, t_mids=[0.4, 0.6, 0.8],
                       t_min=1, t_mins=[0.8, 1, 1.2],
                       # t_min=0, t_mins=None,
                       y_label=r'$v(t)$', fig_size=(9, 3))
