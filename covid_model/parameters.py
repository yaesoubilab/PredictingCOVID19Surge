from enum import Enum

from SimPy.Parameters import Constant, Multinomial, AMultinomialOutcome, Inverse, Division, LinearCombination, \
    Logit, Product, MatrixOfConstantParams, TimeDependentSigmoid, Beta, Uniform, UniformDiscrete
from apace.Inputs import EpiParameters


class Profiles(Enum):
    A = 0   # infected with circulating strain
    B = 1   # infected with novel strain
    # V = 2   # vaccinated and infected with novel strain


class AgeGroups(Enum):
    Age_0_4 = 0
    Age_5_19 = 1
    Age_20_49 = 2
    Age_50_64 = 3
    Age_65_79 = 4
    Age_80_ = 5


class COVIDParameters(EpiParameters):
    """ class to contain the parameters of the COVID model """

    def __init__(self):
        EpiParameters.__init__(self)

        d = 1 / 364  # one day (using year as the unit of time)
        self.nProfiles = len(Profiles)
        self.nAgeGroups = len(AgeGroups)

        # -------- model main parameters -------------
        self.ageDist = Multinomial(par_n=Constant(100000), p_values=[0.060, 0.189, 0.395, 0.192, 0.125, 0.039])
        self.sizeE = UniformDiscrete(minimum=1, maximum=5)
        self.sizeEDist = Multinomial(par_n=self.sizeE, p_values=[0.060, 0.189, 0.395, 0.192, 0.125, 0.039])

        self.sizeS = []
        self.sizeE = []
        for a in range(len(AgeGroups)):
            self.sizeS.append(AMultinomialOutcome(par_multinomial=self.ageDist, outcome_index=a))
            self.sizeE.append([AMultinomialOutcome(par_multinomial=self.sizeEDist, outcome_index=a), Constant(0)])

        self.R0s = [Beta(mean=2.5, st_dev=0.75, minimum=1.5, maximum=4), None]
        self.durE = [Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d),
                     Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d)]
        self.durI = [Beta(mean=4 * d, st_dev=0.5 * d, minimum=2 * d, maximum=8 * d), None]
        self.durHosp = [Beta(mean=12 * d, st_dev=1 * d, minimum=7 * d, maximum=17 * d),
                        Beta(mean=12 * d, st_dev=1 * d, minimum=7 * d, maximum=17 * d)]
        self.durICU = [Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d),
                       Beta(mean=10 * d, st_dev=1 * d, minimum=5 * d, maximum=15 * d)]
        self.durR = [Beta(mean=1, st_dev=0.2, minimum=0.5, maximum=1.5),
                     Beta(mean=1, st_dev=0.2, minimum=0.5, maximum=1.5)]
        self.durVacImmunity = Uniform(0.5, 1.5)  # Beta(mean=1.5, st_dev=0.25, minimum=0.5, maximum=2.5)
        self.importRate = Constant(value=52 * 5)

        self.probHosp = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probICUIfHosp = [Beta(mean=0.326, st_dev=0.018, minimum=0, maximum=1),
                              Beta(mean=0.326, st_dev=0.018, minimum=0, maximum=1)]
        self.probDeathIfICU = [Beta(mean=0.330, st_dev=0.032, minimum=0, maximum=1),
                               Beta(mean=0.330, st_dev=0.032, minimum=0, maximum=1)]
        self.probNovelStrainParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                      Beta(mean=1.25, st_dev=0.2, minimum=0.75, maximum=1.75),  # t0
                                      Uniform(minimum=0.4, maximum=0.6)]  # max
        self.vaccRateParams = [Uniform(minimum=-10, maximum=-6),                 # b
                               Uniform(minimum=1, maximum=1.2),                # t_min
                               Uniform(minimum=1.5, maximum=2),     # t_middle
                               Uniform(minimum=0, maximum=0.5),       # min
                               Uniform(minimum=1.25, maximum=1.75)]     # max

        self.pdY1Thresholds = [UniformDiscrete(2, 10), UniformDiscrete(1, 10)]  # on/off
        self.changeInContactY1 = Uniform(-0.8, -0.5)
        self.changeInContactY1Plus = Uniform(-0.8, -0.5)

        # parameters of the novel strain
        self.ratioTransmmisibilityAToB = Uniform(1, 2)
        self.ratioProbHospAToB = Uniform(0.5, 1.5)
        self.ratioDurInfAToB = Uniform(1, 1.5)

        # calculate dependent parameters
        self.probNovelStrain = None
        self.vaccRate = None
        self.matrixChangeInContactsY1 = None
        self.matrixChangeInContactsY1Plus = None
        self.ratesOfLeavingE = [None] * self.nProfiles
        self.ratesOfLeavingI = [None] * self.nProfiles
        self.ratesOfLeavingHosp = [None] * self.nProfiles
        self.ratesOfLeavingICU = [None] * self.nProfiles
        self.ratesOfLeavingR = [None] * self.nProfiles
        self.infectivity = [None] * self.nProfiles
        self.durInfec = [None] * self.nProfiles
        self.logitProbDeathInICU = [None] * self.nProfiles
        self.ratesOfDeathInICU = [None] * self.nProfiles
        self.rateOfLosingVacImmunity = None

        self.calculate_dependent_params()

        # build the dictionary of parameters
        self.build_dict_of_params()

    def calculate_dependent_params(self):

        # for the novel strain
        self.R0s[1] = Product(parameters=[self.R0s[0], self.ratioTransmmisibilityAToB])
        self.probHosp[1] = Product(parameters=[self.probHosp[0], self.ratioProbHospAToB])
        self.durI[1] = Product(parameters=[self.durI[0], self.ratioDurInfAToB])

        # probability of novel strain
        self.probNovelStrain = TimeDependentSigmoid(
            par_b=self.probNovelStrainParams[0],
            par_t_middle=self.probNovelStrainParams[1],
            par_max=self.probNovelStrainParams[2])

        # vaccination rate
        self.vaccRate = TimeDependentSigmoid(
            par_b=self.vaccRateParams[0],
            par_t_min=self.vaccRateParams[1],
            par_t_middle=self.vaccRateParams[2],
            par_min=self.vaccRateParams[3],
            par_max=self.vaccRateParams[4])

        # change in contact matrices
        self.matrixChangeInContactsY1 = MatrixOfConstantParams([[self.changeInContactY1]])
        self.matrixChangeInContactsY1Plus = MatrixOfConstantParams([[self.changeInContactY1Plus]])

        self.rateOfLosingVacImmunity = Inverse(par=self.durVacImmunity)

        for i in range(2):
            self.ratesOfLeavingE[i] = Inverse(par=self.durE[i])
            self.ratesOfLeavingI[i] = Inverse(par=self.durI[i])
            self.ratesOfLeavingHosp[i] = Inverse(par=self.durHosp[i])
            self.ratesOfLeavingICU[i] = Inverse(par=self.durICU[i])
            self.ratesOfLeavingR[i] = Inverse(par=self.durR[i])

            # duration of infectiousness is the sum of durations in E and I
            self.durInfec[i] = LinearCombination(
                parameters=[self.durE[i], self.durI[i]])

            # infectivity = R0 / (duration of infectiousness)
            self.infectivity[i] = Division(
                par_numerator=self.R0s[i],
                par_denominator=self.durInfec[i])

            # Pr{Death in ICU} = p
            # Rate{Death in ICU} = p/(1-p) * Rate{Leaving ICU}
            self.logitProbDeathInICU[i] = Logit(par=self.probDeathIfICU[i])
            self.ratesOfDeathInICU[i] = Product(parameters=[self.logitProbDeathInICU[i], self.ratesOfLeavingICU[i]])

    def build_dict_of_params(self):
        self.dictOfParams = dict(
            {'Size S': self.sizeS,
             'Size Es': self.sizeE,
             'Size Is': self.sizeI,

             'Transmissibility of B to A': self.ratioTransmmisibilityAToB,
             'Prob of hospitalization of B to A': self.ratioProbHospAToB,
             'Duration of infectiousness of B to A': self.ratioDurInfAToB,

             'Duration of E': self.durE,
             'Duration of I': self.durI,
             'Duration of Hosp': self.durHosp,
             'Duration of ICU': self.durICU,
             'Duration of R': self.durR,
             'Duration of vaccine immunity': self.durVacImmunity,
             'Duration of infectiousness': self.durInfec,

             'R0s': self.R0s,
             'Infectivity': self.infectivity,

             'Rates of leaving E': self.ratesOfLeavingE,
             'Rates of leaving I': self.ratesOfLeavingI,
             'Rates of leaving Hosp': self.ratesOfLeavingHosp,
             'Rates of leaving ICU': self.ratesOfLeavingICU,
             'Rates of leaving R': self.ratesOfLeavingR,

             'Prob Hosp': self.probHosp,
             'Prob ICU | Hosp': self.probICUIfHosp,
             'Prob Death | ICU': self.probDeathIfICU,
             'Logit of prob death in ICU': self.logitProbDeathInICU,
             'Rate of death in ICU': self.ratesOfDeathInICU,

             'Importation rate': self.importRate,
             'Prob novel strain params': self.probNovelStrainParams,
             'Prob novel strain': self.probNovelStrain,
             'Vaccination rate params': self.vaccRateParams,
             'Vaccination rate': self.vaccRate,
             'Rate of losing vaccine immunity': self.rateOfLosingVacImmunity,

             'PD Y1 thresholds': self.pdY1Thresholds,

             'Change in contacts - PD Y1': self.changeInContactY1,
             'Change in contacts - PD Y1+': self.changeInContactY1Plus,
             'Matrix of change in contacts - PD Y1': self.matrixChangeInContactsY1,
             'Matrix of change in contacts - PD Y1+': self.matrixChangeInContactsY1Plus
             })
