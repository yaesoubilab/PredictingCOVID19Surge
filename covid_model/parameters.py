from SimPy.Parameters import Constant, Multinomial, AMultinomialOutcome, Inverse, Division, LinearCombination, \
    Logit, Product, MatrixOfConstantParams, TimeDependentSigmoid, Beta, Uniform, UniformDiscrete
from apace.Inputs import EpiParameters
from definitions import AgeGroups, Profiles


class COVIDParameters(EpiParameters):
    """ class to contain the parameters of the COVID model """

    def __init__(self):
        EpiParameters.__init__(self)

        d = 1 / 364  # one day (using year as the unit of time)
        us_age_dist = [0.060, 0.189, 0.395, 0.192, 0.125, 0.039]
        importation_rate = 52 * 5

        self.nProfiles = len(Profiles)
        self.nAgeGroups = len(AgeGroups)

        # -------- model main parameters -------------
        self.sizeS0 = Constant(100000)
        self.sizeE0 = UniformDiscrete(minimum=1, maximum=5)

        self.distS0ToSs = Multinomial(par_n=self.sizeS0, p_values=us_age_dist)
        self.distE0ToEs = Multinomial(par_n=self.sizeE0, p_values=us_age_dist)

        self.sizeSByAge = []
        self.sizeEProfile0ByAge = []
        self.importRateByAge = []
        for a in range(self.nAgeGroups):
            self.sizeSByAge.append(AMultinomialOutcome(par_multinomial=self.distS0ToSs, outcome_index=a))
            self.sizeEProfile0ByAge.append(AMultinomialOutcome(par_multinomial=self.distE0ToEs, outcome_index=a))
            self.importRateByAge.append(Constant(value=importation_rate * us_age_dist[a]))

        # TODO: figure out R0 by age
        self.R0s = [Beta(mean=2.5, st_dev=0.75, minimum=1.5, maximum=4), None]

        # these duration parameters are age-independent
        self.durEByProfile = [Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d),
                              Beta(mean=5 * d, st_dev=0.5 * d, minimum=1.5 * d, maximum=6 * d)]
        self.durIByProfile = [Beta(mean=4 * d, st_dev=0.5 * d, minimum=2 * d, maximum=8 * d), None]
        self.durHospByProfile = [Beta(mean=12 * d, st_dev=1 * d, minimum=7 * d, maximum=17 * d),
                                 Beta(mean=12 * d, st_dev=1 * d, minimum=7 * d, maximum=17 * d)]
        self.durRByProfile = [Beta(mean=1, st_dev=0.2, minimum=0.5, maximum=1.5),
                              Beta(mean=1, st_dev=0.2, minimum=0.5, maximum=1.5)]
        self.durVacImmunityByProfile = Uniform(0.5, 1.5)  # Beta(mean=1.5, st_dev=0.25, minimum=0.5, maximum=2.5)

        # the probability of hospitalization is assumed to be age- and profile-dependent
        self.probHospByAgeAndProfile = [None] * self.nAgeGroups
        # TODO: update these parameters
        self.probHospByAgeAndProfile[AgeGroups.Age_0_4.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probHospByAgeAndProfile[AgeGroups.Age_5_19.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probHospByAgeAndProfile[AgeGroups.Age_20_49.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probHospByAgeAndProfile[AgeGroups.Age_50_64.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probHospByAgeAndProfile[AgeGroups.Age_65_79.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]
        self.probHospByAgeAndProfile[AgeGroups.Age_80_.value] \
            = [Beta(mean=0.065, st_dev=0.01, minimum=0, maximum=1), None]

        # probability of death if hospitalized is age-independent
        self.probDeathIfHospByProfile = []
        for p in range(self.nProfiles):
            self.probDeathIfHospByProfile.append(Beta(mean=0.1, st_dev=0.032, minimum=0, maximum=1))

        # vaccination rate is age-dependent
        self.vaccRateParams = [Uniform(minimum=-10, maximum=-6),    # b
                               Uniform(minimum=1, maximum=1.2),     # t_min
                               Uniform(minimum=1.5, maximum=2),     # t_middle
                               Uniform(minimum=0, maximum=0.5),     # min
                               Uniform(minimum=1.25, maximum=1.75)] # max

        self.pdY1Thresholds = [Uniform(0, 0.005), Uniform(0, 0.005)]  # on/off
        self.changeInContactY1 = Uniform(-0.75, -0.25)
        self.changeInContactY1Plus = Uniform(-0.75, -0.25)

        # parameters of the novel strain
        self.probNovelStrainParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                      Beta(mean=1.25, st_dev=0.2, minimum=0.75, maximum=1.75),  # t0
                                      Uniform(minimum=0.4, maximum=0.6)]  # max
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
        self.ratesOfLeavingR = [None] * self.nProfiles
        self.infectivity = [None] * self.nProfiles
        self.durInfec = [None] * self.nProfiles
        self.logitProbDeathInHosp = [None] * self.nProfiles
        self.ratesOfDeathInHosp = [None] * self.nProfiles
        self.rateOfLosingVacImmunity = None

        self.calculate_dependent_params()

        # build the dictionary of parameters
        self.build_dict_of_params()

    def calculate_dependent_params(self):

        # for the novel strain
        self.R0s[1] = Product(parameters=[self.R0s[0], self.ratioTransmmisibilityAToB])
        for a in range(self.nAgeGroups):
            self.probHospByAgeAndProfile[a][1] = Product(
                parameters=[self.probHospByAgeAndProfile[a][0], self.ratioProbHospAToB])
        self.durIByProfile[1] = Product(parameters=[self.durIByProfile[0], self.ratioDurInfAToB])

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

        self.rateOfLosingVacImmunity = Inverse(par=self.durVacImmunityByProfile)

        for p in range(self.nProfiles):
            self.ratesOfLeavingE[p] = Inverse(par=self.durEByProfile[p])
            self.ratesOfLeavingI[p] = Inverse(par=self.durIByProfile[p])
            self.ratesOfLeavingHosp[p] = Inverse(par=self.durHospByProfile[p])
            self.ratesOfLeavingR[p] = Inverse(par=self.durRByProfile[p])

            # duration of infectiousness is the sum of durations in E and I
            self.durInfec[p] = LinearCombination(
                parameters=[self.durEByProfile[p], self.durIByProfile[p]])

            # infectivity = R0 / (duration of infectiousness)
            self.infectivity[p] = Division(
                par_numerator=self.R0s[p],
                par_denominator=self.durInfec[p])

            # Pr{Death in Hosp} = p
            # Rate{Death in Hosp} = p/(1-p) * Rate{Leaving Hosp}
            self.logitProbDeathInHosp[p] = Logit(par=self.probDeathIfHospByProfile[p])
            self.ratesOfDeathInHosp[p] = Product(parameters=[self.logitProbDeathInHosp[p], self.ratesOfLeavingHosp[p]])

    def build_dict_of_params(self):
        self.dictOfParams = dict(
            {'Size S0': self.sizeS0,
             'Size E0': self.sizeE0,
             'Distributing S0 to Ss': self.distS0ToSs,
             'Distributing E0 to Es': self.distE0ToEs,
             'Size S by age': self.sizeSByAge,
             'Size E by age': self.sizeEProfile0ByAge,

             'Transmissibility of B to A': self.ratioTransmmisibilityAToB,
             'Prob of hospitalization of B to A': self.ratioProbHospAToB,
             'Duration of infectiousness of B to A': self.ratioDurInfAToB,

             'Duration of E': self.durEByProfile,
             'Duration of I': self.durIByProfile,
             'Duration of Hosp': self.durHospByProfile,
             'Duration of R': self.durRByProfile,
             'Duration of vaccine immunity': self.durVacImmunityByProfile,
             'Duration of infectiousness': self.durInfec,

             'R0s': self.R0s,
             'Infectivity': self.infectivity,

             'Rates of leaving E': self.ratesOfLeavingE,
             'Rates of leaving I': self.ratesOfLeavingI,
             'Rates of leaving Hosp': self.ratesOfLeavingHosp,
             'Rates of leaving R': self.ratesOfLeavingR,

             'Prob Death | Hosp': self.probDeathIfHospByProfile,
             'Logit of prob death in Hosp': self.logitProbDeathInHosp,
             'Rate of death in Hosp': self.ratesOfDeathInHosp,

             'Importation rate': self.importRateByAge,
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

        for a in range(self.nAgeGroups):
            self.dictOfParams['Prob Hosp-age '+str(a)] = self.probHospByAgeAndProfile[a]
