from SimPy.Parameters import Constant, Multinomial, AMultinomialOutcome, Inverse, Division, LinearCombination, \
    Logit, Product, MatrixOfConstantParams, TimeDependentSigmoid, Beta, Uniform, UniformDiscrete, Gamma
from apace.Inputs import EpiParameters
from definitions import AgeGroups, Profiles


class COVIDParameters(EpiParameters):
    """ class to contain the parameters of the COVID model """

    def __init__(self):
        EpiParameters.__init__(self)

        self.nProfiles = len(Profiles)
        self.nAgeGroups = len(AgeGroups)
        d = 1 / 364  # one day (using year as the unit of time)

        # -------- model main parameters -------------
        us_age_dist = [0.060, 0.189, 0.395, 0.192, 0.096, 0.068]
        hosp_relative_risk = [1, 1, 1.7, 4, 6, 10.8]
        prob_death = [0, 0.002, 0.026, 0.079, 0.141, 0.209]
        importation_rate = 52 * 5

        self.sizeS0 = Constant(500000)
        self.sizeE0 = UniformDiscrete(minimum=1, maximum=5)
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
        self.probHosp5To17 = Uniform(0.001, 0.01)

        # vaccination rate is age-dependent
        self.vaccRateParams = [Uniform(minimum=-10, maximum=-5),        # b
                               Uniform(minimum=1.5, maximum=2.5),         # t_middle
                               Uniform(minimum=0.25, maximum=0.5),         # min
                               Uniform(minimum=2, maximum=3)]     # max
        self.vaccRateTMinByAge = [
            Constant(100),                      # 0-4
            Uniform(minimum=1.0, maximum=1.4),  # 5-19
            Uniform(minimum=1.0, maximum=1.4),  # 20-49
            Uniform(minimum=0.9, maximum=1.3),  # 50-64
            Uniform(minimum=0.8, maximum=1.2),  # 65-75
            Uniform(minimum=0.7, maximum=1.1)   # 75+
        ]

        self.pdY1Thresholds = [Uniform(0, 0.0005), Uniform(0, 0.0005)]  # on/off
        self.changeInContactY1 = Uniform(-0.75, -0.25)
        self.changeInContactY1Plus = Uniform(-0.75, -0.25)

        # parameters of the novel strain
        self.probNovelStrainParams = [Beta(mean=7, st_dev=0.5, minimum=5, maximum=9),  # b
                                      Beta(mean=1.75, st_dev=0.1, minimum=1, maximum=2),  # t_middle
                                      Uniform(minimum=0.4, maximum=0.6)]  # max
        self.ratioTransmmisibilityAToB = Uniform(1, 2)
        self.ratioProbHospAToB = Uniform(0.5, 1.5)
        self.ratioDurInfAToB = Uniform(1, 1.5)

        # ------------------------------
        # calculate dependent parameters
        self.sizeSByAge = []
        self.sizeEProfile0ByAge = []
        self.importRateByAge = []
        self.distS0ToSs = None
        self.distE0ToEs = None
        self.probNovelStrain = None
        self.vaccRateByAge = [None] * self.nAgeGroups
        self.matrixChangeInContactsY1 = None
        self.matrixChangeInContactsY1Plus = None
        self.relativeProbHosp = [None] * self.nAgeGroups
        self.probHospByAgeAndProfile = [None] * self.nAgeGroups
        self.probDeathIfHospByAgeAndProfile = [None] * self.nAgeGroups
        self.ratesOfLeavingE = [None] * self.nProfiles
        self.ratesOfLeavingI = [None] * self.nProfiles
        self.ratesOfLeavingHosp = [None] * self.nProfiles
        self.ratesOfLeavingR = [None] * self.nProfiles
        self.infectivity = [None] * self.nProfiles
        self.durInfec = [None] * self.nProfiles
        self.logitProbDeathInHospByAge = [None] * self.nAgeGroups
        self.ratesOfDeathInHospByAge = [None] * self.nAgeGroups
        self.rateOfLosingVacImmunity = None

        self.calculate_dependent_params(us_age_dist=us_age_dist,
                                        hosp_relative_risk=hosp_relative_risk,
                                        prob_death=prob_death,
                                        importation_rate=importation_rate)

        # build the dictionary of parameters
        self.build_dict_of_params()

    def calculate_dependent_params(self, us_age_dist, hosp_relative_risk, prob_death, importation_rate):

        self.distS0ToSs = Multinomial(par_n=self.sizeS0, p_values=us_age_dist)
        self.distE0ToEs = Multinomial(par_n=self.sizeE0, p_values=us_age_dist)

        for a in range(self.nAgeGroups):
            self.sizeSByAge.append(AMultinomialOutcome(par_multinomial=self.distS0ToSs, outcome_index=a))
            self.sizeEProfile0ByAge.append(AMultinomialOutcome(par_multinomial=self.distE0ToEs, outcome_index=a))
            self.importRateByAge.append(Constant(value=importation_rate * us_age_dist[a]))

        # for the novel strain
        self.R0s[1] = Product(parameters=[self.R0s[0], self.ratioTransmmisibilityAToB])

        # relative probability of hospitalization to age 5-17
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_5_19.value:
                self.relativeProbHosp[a] = Constant(1)
            else:
                self.relativeProbHosp[a] = Gamma(mean=hosp_relative_risk[a], st_dev=hosp_relative_risk[a]*0.25)

        # probability of hospitalization by age
        for a in range(self.nAgeGroups):
            self.probHospByAgeAndProfile[a] = [Product(
                parameters=[self.probHosp5To17, self.relativeProbHosp[a]]), None]

        # probability of hospitalization for new variant
        for a in range(self.nAgeGroups):
            self.probHospByAgeAndProfile[a][1] = Product(
                parameters=[self.probHospByAgeAndProfile[a][0], self.ratioProbHospAToB])

        # probability of death by age
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_0_4.value:
                self.probDeathIfHospByAgeAndProfile[a] = [Constant(0), Constant(0)]
            else:
                self.probDeathIfHospByAgeAndProfile[a] = [Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25),
                                                          Beta(mean=prob_death[a], st_dev=prob_death[a]*0.25)]

        self.durIByProfile[1] = Product(parameters=[self.durIByProfile[0], self.ratioDurInfAToB])

        # probability of novel strain
        self.probNovelStrain = TimeDependentSigmoid(
            par_b=self.probNovelStrainParams[0],
            par_t_middle=self.probNovelStrainParams[1],
            par_max=self.probNovelStrainParams[2])

        # vaccination rate
        for a in range(self.nAgeGroups):
            if a == AgeGroups.Age_0_4.value:
                self.vaccRateByAge[a] = Constant(0)
            else:
                self.vaccRateByAge[a] = TimeDependentSigmoid(
                    par_b=self.vaccRateParams[0],
                    par_t_min=self.vaccRateTMinByAge[a],
                    par_t_middle=self.vaccRateParams[1],
                    par_min=self.vaccRateParams[2],
                    par_max=self.vaccRateParams[3])

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

        for a in range(self.nAgeGroups):
            self.logitProbDeathInHospByAge[a] = [None] * self.nProfiles
            self.ratesOfDeathInHospByAge[a] = [None] * self.nProfiles
            for p in range(self.nProfiles):
                # Pr{Death in Hosp} = p
                # Rate{Death in Hosp} = p/(1-p) * Rate{Leaving Hosp}
                self.logitProbDeathInHospByAge[a][p] = Logit(par=self.probDeathIfHospByAgeAndProfile[a][p])
                self.ratesOfDeathInHospByAge[a][p] = Product(
                    parameters=[self.logitProbDeathInHospByAge[a][p], self.ratesOfLeavingHosp[p]])

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

             'Prob Hosp for 5-17': self.probHosp5To17,
             'Relative prob hosp': self.relativeProbHosp,

             'Importation rate': self.importRateByAge,
             'Prob novel strain params': self.probNovelStrainParams,
             'Prob novel strain': self.probNovelStrain,

             'Vaccination rate params': self.vaccRateParams,
             'Vaccination rate t_min by age': self.vaccRateTMinByAge,
             'Vaccination rate': self.vaccRateByAge,
             'Rate of losing vaccine immunity': self.rateOfLosingVacImmunity,

             'PD Y1 thresholds': self.pdY1Thresholds,

             'Change in contacts - PD Y1': self.changeInContactY1,
             'Change in contacts - PD Y1+': self.changeInContactY1Plus,
             'Matrix of change in contacts - PD Y1': self.matrixChangeInContactsY1,
             'Matrix of change in contacts - PD Y1+': self.matrixChangeInContactsY1Plus
             })

        for a in range(self.nAgeGroups):
            self.dictOfParams['Prob Hosp-age '+str(a)] = self.probHospByAgeAndProfile[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Prob Death|Hosp-age' + str(a)] = self.probDeathIfHospByAgeAndProfile[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Logit of prob death in Hosp' + str(a)] = self.logitProbDeathInHospByAge[a]

        for a in range(self.nAgeGroups):
            self.dictOfParams['Rate of death in Hosp' + str(a)] = self.ratesOfDeathInHospByAge[a]