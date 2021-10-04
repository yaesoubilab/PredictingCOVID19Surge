import matplotlib.pyplot as plt
import numpy as np

from SimPy.Parameters import Constant, TimeDependentSigmoid, TimeDependentCosine

SIM_DURATION = 2.25
T0 = 0
X_RANGE = (-0.1, 2.35)
Y_RANGE = (0, 1)
SEPTEMBER_FIRST = 1.5


def sigmoid(t, b, f_min, f_max, t_mid, t_min=0):
    # sigmoid function
    par = TimeDependentSigmoid(
        par_b=Constant(b),
        par_t_min=Constant(t_min),
        par_t_middle=Constant(t_mid),
        par_min=Constant(f_min),
        par_max=Constant(f_max)
    )
    val = par.sample(time=t)
    return val if val > 0 else None


def cos(t, phase, scale, f_min, f_max):
    # cosine function
    par = TimeDependentCosine(
        par_phase=Constant(phase),
        par_scale=Constant(scale),
        par_min=Constant(f_min),
        par_max=Constant(f_max)
    )
    return par.sample(time=t)


def plot_sigmoid_functions(b, f_min, f_max, t_mid, t_min,
                           bs, f_mins, f_maxs, t_mids, t_mins,
                           x_range, y_label, x_label,
                           vertical_line=None, fig_size=(7.5, 3.2), round_b=None):

    # ------------------
    ts = np.linspace(start=x_range[0], stop=x_range[1])
    fs = []
    legends = []
    titles = []

    # varying b
    if bs is not None:
        titles.append('Varying ' + r'$b$')
        legs = []  # legends
        ys = []
        for v in bs:
            if round_b is None:
                legs.append(r'$b=$' + str(v))
            else:
                legs.append(r'$b=$' + str(round(v, round_b)))
            ys.append([sigmoid(t, b=v, t_mid=t_mid, f_min=f_min, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b_min
    if f_mins is not None:
        titles.append('Varying ' + r'$b_{min}$')
        legs = []  # legends
        ys = []
        for v in f_mins:
            legs.append(r'$b_{min}=$' + str(v))
            ys.append([sigmoid(t, b=b, t_mid=t_mid, f_min=v, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying b_max
    if f_maxs is not None:
        titles.append('Varying ' + r'$b_{max}$')
        legs = []  # legends
        ys = []
        for v in f_maxs:
            legs.append(r'$b_{max}=$' + str(v))
            ys.append([sigmoid(t, b=b, t_mid=t_mid, f_min=f_min, f_max=v, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying t_mid
    if t_mids is not None:
        titles.append('Varying ' + r'$t_{mid}$')
        legs = []  # legends
        ys = []
        for v in t_mids:
            legs.append(r'$t_{mid}=$' + str(v))
            ys.append([sigmoid(t, b=b, t_mid=v, f_min=f_min, f_max=f_max, t_min=t_min) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying t_mins
    if t_mins is not None:
        titles.append('Varying ' + r'$t_{min}$')
        legs = []  # legends
        ys = []
        for v in t_mins:
            legs.append(r'$t_{min}=$' + str(v))
            ys.append([sigmoid(t, b=b, t_mid=t_mid, f_min=f_min, f_max=f_max, t_min=v) for t in ts])
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
        if vertical_line is not None:
            ax.axvline(x=vertical_line, c='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel(x_label)
        ax.legend(fontsize='x-small') # loc=2

    axarr[0].set_ylabel(y_label)
    plt.tight_layout()
    plt.show()


def plot_alpha_h(h_max, alpha_max,
                 h_maxs, alpha_maxs,
                 x_range, y_label, x_label,
                 vertical_line=None, fig_size=(7.5, 3.2)):

    # ------------------
    hs = np.linspace(start=x_range[0], stop=x_range[1])
    fs = []
    legends = []
    titles = []

    # varying alpha
    if alpha_maxs is not None:
        titles.append('Varying ' + r'$\bar\alpha$')
        legs = []  # legends
        ys = []
        for v in alpha_maxs:
            legs.append(r'$\bar\alpha=$' + str(v))
            ys.append([sigmoid(h, b=4/h_max, f_min=0, f_max=v, t_mid=0) for h in hs])
        legends.append(legs)
        fs.append(ys)

    # varying h_max
    if h_maxs is not None:
        titles.append('Varying ' + r'$\bar h$')
        legs = []  # legends
        ys = []
        for v in h_maxs:
            legs.append(r'$\bar h=$' + str(v))
            ys.append([sigmoid(h, b=4/v, f_min=0, f_max=alpha_max, t_mid=0) for h in hs])
        legends.append(legs)
        fs.append(ys)

    # plot
    fig, axarr = plt.subplots(1, len(fs), sharey=True, figsize=fig_size)
    for i, ax in enumerate(axarr):
        ax.set_title(titles[i])

        for j in range(3):
            ax.plot(hs, fs[i][j], label=legends[i][j])  # color='b', linestyle='-')

        ax.set_ylim(Y_RANGE)
        ax.set_xlim(X_RANGE)
        if vertical_line is not None:
            ax.axvline(x=vertical_line, c='k', linestyle='--', linewidth=0.5)
        ax.set_xlabel(x_label)
        ax.legend(fontsize='x-small') # loc=2

    axarr[0].set_ylabel(y_label)
    plt.tight_layout()
    plt.show()


def plot_cos_functions(phase, f_min, f_max,
                       phases, f_mins, f_maxes,
                       y_label, fig_size=(7.5, 3.2)):

    # ------------------
    ts = np.linspace(start=T0, stop=SIM_DURATION)
    fs = []
    legends = []
    titles = []

    # varying phases
    if phases is not None:
        titles.append('Varying '+r'$\phi$')
        legs = [] # legends
        ys = []
        for v in phases:
            legs.append(r'$\phi=$'+str(v))
            ys.append([cos(t, phase=v, scale=1, f_min=f_min, f_max=f_max) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying min
    if f_mins is not None:
        titles.append('Varying '+r'$\sigma_{min}$')
        legs = [] # legends
        ys = []
        for v in f_mins:
            legs.append(r'$\sigma_{min}=$'+str(v))
            ys.append([cos(t, phase=phase, scale=1, f_min=v, f_max=f_max) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # varying max
    if f_maxes is not None:
        titles.append('Varying '+r'$\sigma_{max}$')
        legs = [] # legends
        ys = []
        for v in f_maxes:
            legs.append(r'$\sigma_{max}=$'+str(v))
            ys.append([cos(t, phase=phase, scale=1, f_min=f_min, f_max=v) for t in ts])
        legends.append(legs)
        fs.append(ys)

    # plot
    fig, axarr = plt.subplots(1, len(fs), sharey=True, figsize=fig_size)
    for i, ax in enumerate(axarr):
        ax.set_title(titles[i])

        for j in range(len(fs[i])):
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
                       y_label=r'$\gamma(t)$',
                       x_range=[T0, SIM_DURATION],
                       x_label='Simulation Year ' + r'$(t)$',
                       vertical_line=SEPTEMBER_FIRST,
                       fig_size=(5.5, 3))

# rate of vaccination
Y_RANGE = (0, 3)
plot_sigmoid_functions(b=-7.5, bs=[-10, -7.5, -5],
                       f_min=0.1, f_mins=[0, 0.1, 0.2],
                       f_max=2.5, f_maxs=[2, 2.5, 3],
                       t_mid=0.75, t_mids=[0.5, 0.75, 1],
                       t_min=1, t_mins=[0.8, 1, 1.2],
                       # t_min=0, t_mins=None,
                       y_label=r'$v(t)$',
                       x_range=[T0, SIM_DURATION],
                       x_label='Simulation Year ' + r'$(t)$',
                       vertical_line=SEPTEMBER_FIRST,
                       fig_size=(9, 3))

# effectiveness of control strategies
X_RANGE = (-0.1, 30)
Y_RANGE = (0, 1)
MAX_OCC = (10, 15, 20)
plot_alpha_h(h_max=15, alpha_max=0.7,
             h_maxs=[10, 15, 20], alpha_maxs=[0.5, 0.7, 0.9],
             y_label=r'$\alpha(h)$',
             x_range=[0, 40],
             x_label='Hospital occupancy ' + r'$(h)$',
             fig_size=(4.5, 3))

# # seasonality
# Y_RANGE = (0, 3)
# plot_cos_functions(phase=0, phases=[-0.1, 0, 0.1],
#                    f_min=1, f_mins=None,
#                    f_max=2, f_maxes=[1.5, 2, 2.5],
#                    y_label=r'$\sigma(t)$',
#                    x_range=[T0, SIM_DURATION],
#                    x_label='Simulation Year ' + r'$(t)$',
#                    vertical_line = SEPTEMBER_FIRST,
#                    fig_size=(5.3, 3))
