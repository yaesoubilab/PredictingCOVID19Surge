import scipy.stats as scs

import SimPy.FormatFunctions as F
import SimPy.RandomVariateGenerators as Fit

# [mean, stDev, min, max]
R0 = [2.5, 0.7, 1.5, 4]
TimeToInf = [5, 0.5, 3, 7]
TimeInf = [4, 1.5, 2, 8]
DurHosp = [12, 1, 7, 17]
DurICU = [10, 1, 5, 15]
DurR = [1, 0.25, 0.25, 1.5]

ProbHosp = [0.065, 0.010, 0, 1]
ProbICU = [0.326, 0.018, 0, 1]
ProbDeath = [0.330, 0.032, 0, 1]
RatioMortality = [3, 0.25, 1, 5]

# prob{novel strain}
gamma_b = [7, 0.5, 5, 9]
gamma_t0 = [1.25, 0.1, 0.75, 1.75]


def print_intervals(name, mean_std_min_max):
    beta_par = Fit.Beta.fit_mm(
        mean=mean_std_min_max[0],
        st_dev=mean_std_min_max[1],
        minimum=mean_std_min_max[2],
        maximum=mean_std_min_max[3]
    )
    # print(beta_par)
    l = scs.beta.ppf(q=0.025,
                     a=beta_par['a'], b=beta_par['b'], loc=beta_par['loc'], scale=beta_par['scale'])
    u = scs.beta.ppf(q=0.975,
                     a=beta_par['a'], b=beta_par['b'], loc=beta_par['loc'], scale=beta_par['scale'])

    print(name, F.format_interval([l, u], sig_digits=3))


print_intervals('R0:', R0)
print_intervals('Time to infectious:', TimeToInf)
print_intervals('Time infectious:', TimeInf)
print_intervals('Duration of hospitalization:', DurHosp)
print_intervals('Duration of ICU:', DurICU)
print_intervals('Duration of natural immunity:', DurR)
print_intervals('Probability of hospitalization:', ProbHosp)
print_intervals('Probability of ICU:', ProbICU)
print_intervals('Probability of death if needing ICU:', ProbDeath)
print_intervals('Ratio mortality while waiting for ICU', RatioMortality)

print_intervals('Gamma-b', gamma_b)
print_intervals('Gamma-t0', gamma_t0)
