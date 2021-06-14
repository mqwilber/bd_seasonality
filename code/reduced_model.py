import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fmin, brentq
import reduced_mcmc as mcmc
import yaml
import scipy.stats as stats
from scipy.linalg import block_diag

"""
Reduced host-parasite IPM

State variables for each species

L(t): Larvae (tadpoles) at time t
S(t): Susceptible, post-metamorphic hosts at time t
I(t): Infected post-metamorphic hosts at time t
P(t): Total parasites on the log scale at time t
Z(t): Zoospore density in the community at time t

"""

class Community(object):
    """
    The community amphibian Bd model from the reduced IPM

    Required parameters in spp_params
    ---------------------------------

    's0': Uninfected survival probability in a time step
          Range: Between 0 and 1
    'sI': Proportion infection reduces survival in a time step
          Range: Between 0 and 1
    'mu_s': LD50 for load-dependent survival probability (in log load units)
            Load-dependent survival probability follows a Probit function.
            Range: All real numbers
    'sigma_s': Shape parameter for load-dependent survival probability
               Range: All positive numbers
    'lI': Loss of infection probability on the logit scale (independent of load)
          Range: All real numbers
    'lI_temp': Effect of temperature on loss of infection =
               Range: All real numbers
    'mu_l': Log load at which there is a 50% chance of losing infection in a time step
            Loss of infection probability follows a Probit function.
            Range: All real numbers
    'sigma_l': Shape parameter for the loss of infection function
               Range: All positive numbers
    'trans_beta': Transmission parameter. 1 - exp(-trans_beta * Zoospores) gives
                  the probability of infection in a time step.
                  Range: All positive numbers
    'trans_beta_temp': Effect of temperature on transmission parameter
                       Range: All real numbers
    'mu0': Mean initial log infection load
           Range: All real numbers
    'mu0_temp': Effect of temperature on initial infection. Only relevant if fit_mu0 = True
                Range: All real numbers
    'fit_mu0': Bool specifying whether or not to fit initial infection. Default is False
               Values: True or False
    'a': Log pathogen growth rate
         Range : All real numbers but typically greater than 0
    'a_temp': Effect of temperature on log pathogen growth rate
              Range: All real numbers
    'b': Density-dependence in parasite growth
         Range: Between 0 and 1
    'sigma_full': Standard deviation in log pathogen load on hosts in population
                  Range: All positive numbers
    'sigma_full_temp': Temperature-dependence in the standard deviation
                       Range: All real numbers
    'lam': Per pathogen shedding rate
           Range : All positive numbers
    'r': Reproductive rate
         Range : All positive numbers
    'K': Density-dependence parameter (on the log scale)
         Range : All real numbers
    'mean_temp': The mean temperature against which all temperatures are differenced
                 Range : Temperature is in celsius
    'breeding_start': Day in year when breeding starts (e.g. 120)
                      Range : 0 to 365
    'breeding_end': Day in year when breeding ends (e.g. 250)
                    Range : 0 to 365. If less than breeding start, wraps around the year
    'aquatic': 0 or 1 specifying whether or not an amphibian is fully aquatic
               Values: 0 or 1
    'tad_shedding': Per tadpole shedding rate of zoospores
                    Range: All positive numbers
    'larval_period': Length in days of the average larval period
                     Range: All positive numbers
    'omega': Log external contribution to the zoospore pool per time step
             Range : All real numbers
    'hibernation' : Does the individual hibernate? (0) No or (1) Yes
                    Values: 0 or 1
    'hibernation_aquatic' : Does hibernation happen in water where transmission can occur? 0 (No) or 1 (Yes)
                            Values: 0 or 1
    'hibernation_start' : When does the animal start to hibernate (int in days)
                          Range: 0 to 365
    'hibernation_end' : When does the animal stop hibernating (int in days)
                        Range: 0 to 365

    """

    def __init__(self, spp_params, z_fxn, init, model_params, area=1.0,
                 init_temp=None, load_dependent=False, year_length=365,
                 tad_survival=0.2, init_from_equil=True):
        """
        Initialize the community model for each species

        Parameters
        ----------
        spp_params : dict
            Keywords are species names and each species must contain the
            parameters listed above.
        z_fxn : fxn
            Zoospore survival probability as a function of temperature
        init : dict
            Dictionary with keywords being species names and one keyword 'Z'
            that gives how many zoospores are in the pool.
        model_params : dict
            Basic parameters for the model. Holds model level parameters.
            'time_step' : The length of a time_step
        area : float
            The area of the community. Defaults to 1
        load_dependent : bool
            Whether or not loss of infection depends on load
        year_length : int
            The length in days of a year.
        tad_survival : float
            Assume a tad_survival_prob over length of tadpole stage. This is
            fixed at 0.2.
        init_from_equil : bool
            If True set the initial starting point from approximate disease
            free equilibrium

        """

        # Identify how many species are in the community
        self.spps = list(spp_params.keys())
        self.numspp = len(self.spps)
        self.area = area
        self.model_params = model_params
        self.year_length = year_length

        if init_temp is None:
            self.init_temp = model_params['mean_temp']
        else:
            self.init_temp = init_temp

        # Derived parameters
        for spp in self.spps:
            spp_params[spp]['repro_time'] = get_repro_time(spp_params[spp]['breeding_start'],
                                                           spp_params[spp]['breeding_end'],
                                                           self.year_length)
            t = (spp_params[spp]['larval_period'] / model_params['time_step'])
            spp_params[spp]['meta_prob'] = (1.0 / t)

            # Assume only 20% survival period over average larval period
            spp_params[spp]['s_tadpole'] = (0.2)**(1.0 / t)

        self.params_dict = spp_params

        # Set initial values based on model equilibrium values
        if init_from_equil:
            initZ = 0
            for spp in self.spps:
                t_tad, t_adult = adult_tad_equil(self.params_dict[spp]['s0'],
                                                 self.params_dict[spp]['s_tadpole'],
                                                 self.params_dict[spp]['meta_prob'],
                                                 self.params_dict[spp]['r'] / (self.year_length / self.model_params['time_step']),
                                                 np.exp(self.params_dict[spp]['K'] / self.area))

                _, a, mu0, lI, sigma_full = self.temperature_effects(self.params_dict[spp],
                                                         self.init_temp, 0, 0, 0)

                # Just getting approximate initial conditions, so don't worry about load-dependent loss
                if load_dependent:
                    lI = lI

                sl = self.params_dict[spp]['s0']*self.params_dict[spp]['sI']*(1 - lI)
                mean_load = (sl*a + (1 - sl)*mu0) / (1 - self.params_dict[spp]['b']*sl)
                initN = t_adult / 2.0
                initZ += (initN * np.exp(mean_load) * np.exp(sigma_full**2 / 2.0))
                init[spp][0:4] = np.array([t_tad, initN, initN, initN*mean_load])

            self.init = np.concatenate([init[spp] for spp in self.spps])
            self.init = np.concatenate([self.init, np.array([initZ / (1 - z_fxn(self.init_temp))])])
        else:
            self.init = np.concatenate([init[spp] for spp in self.spps + ['Z']])

        # Species-specific params
        self.tad_shedding = np.array([spp_params[spp]['tad_shedding'] for spp in self.spps])
        self.lams = np.array([spp_params[spp]['lam'] for spp in self.spps])
        self.sigmas = np.array([spp_params[spp]['sigma_full'] for spp in self.spps])
        self.sigmas_temp = np.array([spp_params[spp]['sigma_full_temp'] for spp in self.spps])
        self.aquatic = np.array([spp_params[spp]['aquatic'] for spp in self.spps]).astype(np.bool)
        self.bstart = np.array([spp_params[spp]['breeding_start'] for spp in self.spps])
        self.bend = np.array([spp_params[spp]['breeding_end'] for spp in self.spps])
        self.hibernation = np.array([spp_params[spp]['hibernation'] for spp in self.spps]).astype(np.bool)
        self.hibernation_aquatic = np.array([spp_params[spp]['hibernation_aquatic'] for spp in self.spps]).astype(np.bool)
        self.hstart = np.array([spp_params[spp]['hibernation_start'] for spp in self.spps])
        self.hend = np.array([spp_params[spp]['hibernation_end'] for spp in self.spps])

        # External contribution to zoospore pool
        # NOTE: This shouldn't be species specific, but OK if you are just fitting one species
        self.omegas = np.array([spp_params[spp]['omega'] for spp in self.spps])
        self.omega_temps = np.array([spp_params[spp]['omega_temp'] for spp in self.spps])

        # Account for breeding spanning years
        self.wrap = self.bstart > self.bend
        new_starts = self.bend[self.wrap]
        new_ends = self.bstart[self.wrap]
        self.bstart[self.wrap] = new_starts
        self.bend[self.wrap] = new_ends

        # Account for hibernation spanning years
        self.hwrap = self.hstart > self.hend
        new_hstarts = self.hend[self.hwrap]
        new_hends = self.hstart[self.hwrap]
        self.hstart[self.hwrap] = new_hstarts
        self.hend[self.hwrap] = new_hends

        self.time_step = model_params['time_step']

        # Build temperature-dependent zoospore survival function
        self.zsurv_fxn = z_fxn
        self.load_dependent = load_dependent

    def simulate(self, steps, temp_fxn=None, temp_fxn_cv=None, temp_fxn_mm=None):
        """
        Iterate the reduced IPM forward in time.

        Parameters
        ----------
        steps : int
            Number of steps to project the model forward
        temp_fxn : function
            A function that takes year time and returns temperature

        Returns
        -------
        res : array
            Dimensions (4*species) + 1 x steps, containing
            all data from all species and a zoospore pool.
        """

        # Set-up the temperature function
        time = np.arange(steps)
        res = np.zeros((4*self.numspp + 1, len(time)))
        res[:, 0] = np.array(self.init)

        # Default is no temperature effect
        if temp_fxn is None:
            temp_fxn = lambda x: 0

        if temp_fxn_cv is None:
            temp_fxn_cv = lambda x: 0

        if temp_fxn_mm is None:
            temp_fxn_mm = lambda x: 0

        for t in time[1:]:

            val_now = res[:, (t - 1)]

            year_time = t*self.time_step
            temp = temp_fxn(year_time)
            temp_cv = temp_fxn_cv(year_time)
            mismatch = temp_fxn_mm(year_time)

            # Build matrix and update for density-dependence
            Ks = [self.build_K_matrix_log(val_now[(4*i):(4*i + 4)],
                                      val_now[-1], temp, temp_cv, mismatch, t, i,
                                      self.params_dict[spp], self.load_dependent)
                  for i, spp in enumerate(self.spps)]
            # K = block_diag(*Ks)
            # K = self.add_Z_log(K, temp, t, val_now[:-1])

            Zrow = self.add_Z_log(None, temp, t, val_now[:-1])

            Fs = [self.build_F_matrix(val_now[(4*i):(4*i + 4)],
                                      t,
                                      self.params_dict[spp])
                  for i, spp in enumerate(self.spps)]
            # F = block_diag(*Fs)
            # F = add_zeros(F)

            val_next1 = np.concatenate([np.dot(Fs[i] + Ks[i], val_now[(4*i):(4*i + 4)])
                                        for i, spp in enumerate(self.spps)])
            val_next = np.r_[val_next1, np.dot(Zrow, val_now)]

            # Project and include external contribution of zpool
            base_temp = temp - self.model_params['mean_temp']
            # val_next = np.dot(F + K, val_now)
            val_next[-1] = val_next[-1] + np.sum(np.exp(self.omegas + self.omega_temps*base_temp)) / self.area
            res[:, t] = val_next

        return(res)

    def build_K_matrix_log(self, svar, Z, temp, temp_cv, mismatch, time,
                           index, params, load_dependent=False):
        """
        Build fundamental transition matrix for a particular species in the
        model.  Pathogen load on the log scale.

        Parameters
        ----------
        svar : array
            Length 3: S, I, P state variables for a particular species
        Z : float
            The total zoospores
        temp : float
            Temperature
        temp_cv : float
            Coefficient in variation in temperature
        mismatch : float
            Thermal mismatch with historical temperature
        time : float
            Model time
        index : int
            Index of species
        params : dict
            Species-specific parameters

        Returns
        -------
        K : 4 x 4 array
            Transition matrix
        """

        I = svar[2]
        P = svar[3]
        N = svar[1] + I

        # Update parameters with temperature effects
        beta, a, mu0, lI, sigma_full = self.temperature_effects(params, temp, temp_cv,
                                                    mismatch, Z)

        # Update transmission parameter if time in breeding range.
        inpool = self.in_pool(time)[0]

        if not inpool:
            beta = 0  # Assuming when host is not in water it does not experience any force of infection

        # Density dependent transmission.
        phi = 1 - np.exp(-beta*Z)

        # Density-dependent recruitment
        dd_recruit = np.exp(-(np.exp(params['K']) / self.area)*N)

        # Build matrix
        colT = np.array([params['s_tadpole']*(1 - params['meta_prob']),
                         params['s_tadpole']*params['meta_prob']*dd_recruit,
                         0,
                         0])

        colS = np.array([0,
                         params['s0']*(1 - phi),
                         params['s0']*phi,
                         params['s0']*mu0*phi])

        if load_dependent:
            # Load-dependent loss of infection

            # Build coefficients for load-dependent loss of infection
            if I == 0:
                talpha = np.inf
            else:
                talpha = ((P / I) - params['mu_l']) / np.sqrt(params['sigma_l']**2 + sigma_full**2)

            tzeta = sigma_full**2 / np.sqrt(params['sigma_l']**2 + sigma_full**2)
            lI_prime = 1 - stats.norm.cdf(talpha)
            tphi = stats.norm.pdf(talpha)


            colI = np.array([0,
                             params['s0']*lI_prime*params['sI'],
                             params['s0']*(1 - lI_prime)*params['sI'],
                             a*params['sI']*params['s0']*(1 - lI_prime) + params['b']*params['s0']*params['sI']*tzeta*tphi])

            colP = np.array([0,
                             0,
                             0,
                             params['b']*params['sI']*params['s0']*(1 - lI_prime)])

        else:
            # Load-independent loss of infection
            colI = np.array([0,
                             params['s0']*params['sI']*lI,
                             params['s0']*params['sI']*(1 - lI),
                             a*params['s0']*params['sI']*(1 - lI)])
            colP = np.array([0,
                             0,
                             0,
                             params['b']*params['sI']*params['s0']*(1 - lI)])


        return(np.vstack([colT, colS, colI, colP]).T)

    def temperature_effects(self, params, temp, temp_cv, mismatch, Z):
        """
        Update parameters with temperature effects

        Parameters
        ----------
        params : dict
            Species specific parameters
        temp : float
            Temperature
        temp_cv : float
            Coefficient of variation in temperature
        mismatch : float
            Mismatch between current and historical temperature
        Z: float
            Zoospores in pool

        Returns
        -------
        : tuple
            (beta (transmission), a (pathogen growth),
             mu0 (initial infection), lI (loss of infection))
        """

        # Temperature effects
        base_temp = temp - self.model_params['mean_temp']

        # Transmission
        linbeta = (base_temp*params['trans_beta_temp'])

        #
        beta = ((params['trans_beta'] / self.area) *
                np.exp(linbeta)) # m^2 / 7 days = 1 time step

        # Growth
        a = (params['a'] +
             params['a_temp']*base_temp)

        # Initial infection. Saturating infection function
        if params['fit_mu0']:
            mu0lin = params['mu0'] + params['mu0_mm']*mismatch + params['mu0_temp']*base_temp
            mu0 = mu0lin
        else:
            mu0 = a # You get a small infection load early on and then that load experiences growth.

        # Loss of infection: This is irrelevant if load_dependent = True
        linlI = (params['lI'] +
                 params['lI_temp']*base_temp)

        lI = 1 / (1 + np.exp(-(linlI)))

        # Temperature effects on variance.
        sigma_full = params['sigma_full']*np.exp(params['sigma_full_temp']*base_temp)

        return((beta, a, mu0, lI, sigma_full))

    def add_Z_log(self, K, temp, time, vals):
        """
        Add a single row for (Z)oospores. Pathogen load on the log scale.

        Parameters
        ----------
        K : array
            Full transition matrix without zoospores
        temp : float
            Temperature
        time : int
            Model time
        vals : array-like
            All state variables excluding Z

        Returns
        -------
        : array-like
            The final row of the reduced matrix

        """

        # Temperature-dependent zoospore survival
        # nu = 1 / (1 + np.exp(-(self.z_params['nu']
        #                        + temp*self.z_params['nu_temp'])))
        base_temp = temp - self.model_params['mean_temp']
        nu = self.zsurv_fxn(temp)
        sip = vals.reshape((self.numspp, 4))
        Ps = sip[:, 3]
        Is = sip[:, 2]

        # Update shedding parameters if hosts in pool.  Account for temperature-dependence
        sigma_fulls = np.array([self.sigmas*np.exp(base_temp*self.sigmas_temp)
                                for spp in self.spps])
        lams = (self.lams*np.exp(Ps / Is)*np.exp(0.5*sigma_fulls**2))
        lams[np.isnan(lams)] = 0  # if Is has 0s

        # Accounts for hibernation and breeding
        notinpool = ~self.in_pool(time)
        lams[notinpool] = 0  # No shedding when amphibian not in pool

        # Species-specific shedding rates
        lam_mat = np.vstack([self.tad_shedding,
                             np.zeros(self.numspp),
                             lams,
                             np.zeros(self.numspp)]).T.ravel()

        Zrow = np.concatenate([lam_mat, [nu]])
        return(Zrow)

    def build_F_matrix(self, svars, time, params):
        """
        Build the reproduction matrix F for specific species. Accounts
        for density-dependence.

        Parameters
        ----------
        svars : array-like
            Length 4: T, S, I, P
        time : float
            Model time step
        params : dict
            Species-specific parameters

        Returns
        -------
        : array
            4 x 4 reproduction matrix
        """

        F = np.zeros((4, 4))

        year_time = (time*self.time_step) % self.year_length
        lower = year_time - self.time_step
        repro_ind = ((params['repro_time'] > lower) &
                     (params['repro_time'] <= year_time))
        if repro_ind:
            F[0, :] = np.array([0,
                                params['r'],
                                params['r'],
                                0])

        return(F)

    def spp_R0(self, spp, abund, temp, approx_mstar=None):
        """
        Temperature-dependent species R0

        Parameters
        ----------
        spp : str
            Species name
        abund : float
            Species abundance/density
        temp : float
            Temperature
        approx_mstar : None or float
            If a float, this is the value that is used Mstar

        Returns
        -------
        : float
            Species-level R0
        """

        # Get temperature-dependent parameters
        tp = self.params_dict[spp]
        beta, a, mu0, lI, sigma_full = self.temperature_effects(tp, temp, 0, 0, 0)

        part1 = (abund*tp['s0']*beta) / (1 - self.zsurv_fxn(temp))

        # Load dependent loss of infection
        if self.load_dependent:
            tparams = tp.copy()
            tparams['beta'] = beta
            tparams['a'] = a
            tparams['mu0'] = mu0
            M1star = mstar_load_dependent_loss(tparams)

            alpha = (M1star - tp['mu_l']) / np.sqrt(sigma_full**2 + tp['sigma_l']**2)
            lI = stats.norm.sf(alpha)
            sl = tp['sI']*tp['s0']*(1 - lI)
        else:
            sl = tp['sI']*(1 - lI)*tp['s0']
            M1star = (a*(sl) + (1 - sl)*mu0) / (1 - tp['b']*sl)

        part2 = (np.exp(M1star)*np.exp(sigma_full**2 / 2) * tp['lam']) / (1 - sl)
        R0 = part1 * part2
        return(R0)

    def in_pool(self, time):
        """
        For a given time of year, check whether species is in the pool.

        Accounts for hibernation and breeding.

        Returns
        -------
        : Array of bools specifying whether or not species in community are in
          pool at time t.

        """

        year_time = (time*self.time_step) % self.year_length
        breeding = np.bitwise_xor(np.bitwise_and(year_time >= self.bstart,
                                  year_time <= self.bend), self.wrap)
        inpool = ~np.bitwise_and(~self.aquatic, ~breeding).astype(np.bool)

        # Is the time right for hibernation?
        hibernating = np.bitwise_xor(np.bitwise_and(year_time >= self.hstart,
                                     year_time <= self.hend), self.hwrap)

        # Does the animal even hibernate?
        hibernating = np.bitwise_and(self.hibernation, hibernating)

        # If the animal is hibernating, is it in the water?
        hibernating_in_water = np.bitwise_and(self.hibernation_aquatic, hibernating)

        # Is the animal either hibernating in the water or breeding in the pool?
        inpool = np.bitwise_or(inpool, hibernating_in_water)

        return(inpool)


def muF_log_implicit(M, params):
    """
    Function for implicitly solving for mean pathogen load when survival
    depends on pathogen load (log scale)
    """

    a = params['a']
    b = params['b']
    mu0 = params['mu0']
    muS = params['mu_s']
    sigmaS = params['sigma_s']
    sigmaF = params['sigma_full']
    lI = params['lI']
    s0 = params['s0']
    zeta = sigmaF / sigmaS
    alpha = (M - muS) / sigmaS
    gamma = np.sqrt(1 + zeta**2)
    arg1 = alpha / gamma
    arg2 = zeta / np.sqrt(1 + zeta**2)
    norm = stats.norm(loc=0, scale=1)
    sI = norm.sf(arg1)

    return(sI*(1 - lI)*s0*(a + b*M) - (1 - lI)*s0*b*sigmaF*arg2*norm.pdf(arg1) + (1 - sI*(1 - lI)*s0)*mu0 - M)


def mstar_load_dependent(params):
    """
    Predicted individual-level mean when survival depends on pathogen load

    Parameters
    ----------
    params : dict
        Keywords: a, b, mu0, mu_s, sigma_s, lI, sigma_full, s0
    """

    a = params['a']
    b = params['b']
    lower = params['mu0']

    if b != 1:
        upper = a / (1 - b)
    else:
        upper = 100*params['mu_s']

    if lower == upper:
        lower = lower - 0.5
        upper = upper + 0.5

    try:
        equil = brentq(muF_log_implicit, lower, upper, args=(params))
    except ValueError:
        equil = brentq(muF_log_implicit, lower - 2.0, upper, args=(params))

    return(equil)


def muF_log_implicit_loss(M, params):
    """
    Function for implicitly solving for mean pathogen load when loss
    depends on pathogen load (log scale)
    """

    a = params['a']
    b = params['b']
    mu0 = params['mu0']
    mul = params['mu_l']
    sigmal = params['sigma_l']
    sigmaF = params['sigma_full']
    s0 = params['s0']
    alpha = (M - mul) / np.sqrt(sigmaF**2 + sigmal**2)
    zeta = sigmaF**2 / np.sqrt(sigmaF**2 + sigmal**2)
    lI = stats.norm.sf(alpha)
    sI = params['sI']

    return(sI*(1 - lI)*s0*(a + b*M + (b*zeta*stats.norm.pdf(alpha) / (1 - lI))) + (1 - sI*(1 - lI)*s0)*mu0 - M)


def mstar_load_dependent_loss(params):
    """
    Predicted individual-level mean when loss of infection depends on pathogen load

    Parameters
    ----------
    params : dict
        Keywords: a, b, mu0, mu_s, sigma_s, lI, sigma_full, s0
    """

    a = params['a']
    b = params['b']
    lower = a

    if b != 1:
        upper = a / (1 - b)
    else:
        upper = 1000#*params['mu_s']

    if lower == upper:
        lower = lower - 0.5
        upper = upper + 0.5

    try:
        equil = brentq(muF_log_implicit_loss, lower, upper, args=(params))
    except ValueError:
        # What is happening here is that when sigma_full > 0 and mean load
        # is low, individuals with high load disproportionately maintain
        # infection such that observed load is actually higher than a / (1 - b)
        equil = brentq(muF_log_implicit_loss, lower, upper + 20, args=(params))

    return(equil)


def adult_tad_equil(s0, s_tadpole, meta_prob, r, K):
    """
    Equilibrium adult and tadpole densities based on model with
    continuous reproduction. Used to set initial values

    Parameters
    ----------
    s0 : float
        Adult survival prob in a time step
    s_tadpole : float
        Tadpole survival prob in a time step
    meta_prob : float
        Probability of metamorphosis in a time step
    r : float
        Host reproductive rate in a time step (assuming continuous reproduction)
    K : float
        Density-dependence (high is more density-dependence)

    Returns
    -------
    : tuple
        (tadpole equilibrium, adult equilibrium)

    """

    frac = (((1 - s0)*(1 - s_tadpole*(1 - meta_prob)))
            / (r*s_tadpole*meta_prob))
    a_equil = -np.log(frac) / K
    tad_equil = ((a_equil * r)
                 / (1 - s_tadpole*(1 - meta_prob)))
    return((tad_equil, a_equil))


def zsurv_fxn(time_step):
    """
    Temperature-dependent zoospore survival

    Parameters
    ----------
    time_step: int
        Time-step in days of the IPM model

    Returns
    -------
    : Interpolated temperature-dependent zoospore survival function.

    """

    # Observed Zoospore survival from Woodhams et al. 2007
    temp_vals = np.array([4.0, 14.5, 23.0, 28.0])
    drates = np.array([0.0027, 0.0035, 0.0160, 0.0410])
    obs_surv = np.exp(-drates*24*time_step)
    maxsurv = np.max(obs_surv)

    # Minimize sum of squares
    def min_distance(args, x, y, maxval):
        ld50, s = args
        predy = (1 / (1 + np.exp((x - ld50) / s)))*maxval
        ss2 = np.sum((y - predy)**2)
        return(ss2)

    ld50, s = fmin(min_distance, (20, 1), args=(temp_vals, obs_surv, maxsurv),
                   disp=False)

    # This is faster function than interpolation
    def intp(x):
        return((1 / (1 + np.exp((x - ld50) / s)))*maxsurv)

    # intp = interp1d(temp_vals, obs_surv, kind="linear", bounds_error=False,
    #                 fill_value=(obs_surv[0], obs_surv[-1]))

    return(intp)


def add_zeros(F):
    """ Pad matrix F with zeros """

    rows, cols = F.shape
    zrow = np.zeros(rows)[:, np.newaxis]
    zcol = np.zeros(cols + 1)
    return(np.vstack([np.hstack([F, zrow]), zcol]))


def get_repro_time(start, end, year_length=365):
    """
    Calculate reproduction time as midway between start and end of breeding
    period

    Accounts for wrapping of year

    Parameters
    ----------
    start : float
        Start of breeding
    end : float
        End of breeding
    """

    blen = end - start
    if blen < 0:
        blen = year_length - (start - end)

    mid = blen / 2
    repro_time = start + mid

    if repro_time > year_length:
        repro_time = repro_time - year_length

    return(repro_time)


def buildF(p):
    """
    Linearized (F)ecundity matrix for the reduced IPM

    Parameters
    ----------
    p : dict
        Contains keywords: S, beta, s0, inpool, mu0

    Returns
    -------
    : array-like
    """

    val1 = p['S']*p['beta']*p['s0']*p['inpool']
    F = np.matrix([[0, 0, val1],
                   [0, 0, val1*p['mu0']],
                   [0, 0, 0]])
    return(F)


def buildM(p):
    """
    Linearized M matrix for the reduced IPM with load-independent survival

    Parameters
    ----------
    p : dict
        Contains keywords: sI, lI, a, b, sigma_full, lam, Mstar, inpool, zsurv

    """

    sl = p['sI'] * (1 - p['lI']) * p['s0']
    M = np.matrix([[sl,        0,         0],
                   [p['a']*sl, p['b']*sl, 0],
                   [np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*(1 - p['Mstar'])*p['inpool'],
                    np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*p['inpool'],
                    p['zsurv']]])
    return(M)


def buildM_loss_dep(p):
    """
    Linearized M matrix for the reduced IPM with load-dependent loss of infection

    Parameters
    ----------
    p : dict
        Contains keywords: mu_l, sigma_l, sI, a, b, sigma_full,
                           lam, Mstar, inpool, zsurv, s0

    """

    denom = np.sqrt(p['sigma_l']**2 + p['sigma_full']**2)
    alpha = (p['Mstar'] - p['mu_l']) / denom
    lM = 1 - stats.norm.cdf(alpha)
    phiM = stats.norm.pdf(alpha)
    zeta = p['sigma_full']**2 / denom
    surv = p['s0']*p['sI']

    # The linearized reduced IPM with load-dependent loss of infection
    a11 = surv*((1 - lM) - (phiM*p['Mstar']) / denom)
    a12 = (surv * phiM) / denom
    a21 = surv*(p['a']*(1 - lM)
                - ((p['a']*p['Mstar']*phiM) / (denom))
                + p['b']*zeta*phiM
                + ((p['b']*zeta*p['Mstar']*alpha*phiM) / denom)
                - ((p['Mstar']**2 * p['b']*phiM) / denom))
    a22 = surv*(p['a']*phiM / denom
                - ((p['b']*zeta*alpha*phiM) / denom)
                + p['b']*(1 - lM)
                + ((p['Mstar']*p['b']*phiM) / denom))

    a31 = np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*(1 - p['Mstar'])*p['inpool']
    a32 = np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*p['inpool']

    M = np.matrix([[a11, a12, 0],
                   [a21, a22, 0],
                   [a31, a32, p['zsurv']]])
    return(M)


def buildM_surv_dep(p):
    """
    Linearized M matrix for the reduced IPM with load-dependent survival

    Parameters
    ----------
    p : dict
        Contains keywords: mu_s, sigma_s lI, a, b, sigma_full,
                           lam, Mstar, inpool, zsurv, s0

    """

    denom = np.sqrt(p['sigma_s']**2 + p['sigma_full']**2)
    alpha = (p['Mstar'] - p['mu_s']) / denom
    sM = 1 - stats.norm.cdf(alpha)
    phiM = stats.norm.pdf(alpha)
    zeta = p['sigma_full']**2 / denom
    noloss = (1 - p['lI'])*p['s0']

    # The linearized reduced IPM with load-dependent mortality
    a11 = (noloss)*(sM + (phiM*p['Mstar']) / denom)
    a12 = (-(noloss) * phiM) / denom
    a21 = (noloss)*(p['a']*(sM + (phiM*p['Mstar']) / denom)
                    - p['b']*zeta*(phiM + (alpha*phiM*p['Mstar']) / denom)
                    + (phiM*p['Mstar']**2) / denom)
    a22 = (noloss)*((-p['a']*phiM) / denom
                    + (p['b']*zeta*alpha*phiM) / denom
                    + p['b']*(sM - (phiM*p['Mstar']) / denom))
    a31 = np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*(1 - p['Mstar'])*p['inpool']
    a32 = np.exp(p['sigma_full']**2 / 2)*p['lam']*np.exp(p['Mstar'])*p['inpool']

    M = np.matrix([[a11, a12, 0],
                   [a21, a22, 0],
                   [a31, a32, p['zsurv']]])
    return(M)


def block_diag_offset(arrays):
    """
    Make a special block diagonal matrix on lower, off diagonal

    Parameters
    ----------
    arrays : array-like
        List/array of square matrices

    Returns
    -------
    : matrix

    Example
    -------

    arrays = [A, B, C] where A, B, C are square matrices

    R =[0 0 C
        A 0 0
        0 B 0]

    R is returned

    """

    Afinal = arrays[-1]
    As = arrays[:-1]

    n = Afinal.shape[0]
    p = len(arrays)
    n_full = n * p
    fullA = np.zeros((n_full, n_full))

    subA = block_diag(*As)
    fullA[:n, (n*p - n):(n*p)] = Afinal
    fullA[n:, :(n*p - n)] = subA
    return(fullA)


def distance_metric(obs, pred, weights=None):
    """
    Euclidean distance metric

    Parameters
    ----------
    obs : array-like
        Observed vector
    pred : array-like
        Predicted vector
    weight : array-like or None
        If None all observations are weighted equally. Otherwise, uses given
        weights and normalize if necessary.

    Returns
    -------
    : weighted Euclidean distance

    """

    if weights is None:
        weights = np.repeat(1, len(obs)) / len(obs)

    # Normalize if necessary
    if np.sum(weights) != 1:
        weights = weights / np.sum(weights)

    obs = np.array(obs)
    pred = np.array(pred)

    mean_obs = np.nanmean(obs)

    # Standardize
    obsz = (obs - mean_obs) / mean_obs
    predz = (pred - mean_obs) / mean_obs

    return(np.sqrt(np.nansum(weights*(obsz - predz)**2)))


def format_simulation(all_res, time_vals,
                      start_date=pd.datetime(2010, 1, 1)):
    """
    Get summary statistics from the IPM simulation and put them in a
    date formatted  DataFrame.  Used to match simulation data and
    observed data.

    Parameters
    ----------
    all_res : Matrix
        S x N matrix where S is the number of stages and N is number of simulations
        (Tadpoles Sus + Tadpole Inf + Adult Terrestrial + Adult Susceptible + Infecteds + Zoospores)
    time_vals : array-like
        Integers that will specify time since the start date.
    start_date : DateTime object
        The starting date of the simulation. Default date given.


    """

    dates = start_date + pd.to_timedelta(time_vals, unit="Days")

    mean_loads = all_res[2, :] / all_res[1, :]
    num_uninf = all_res[0, :]
    num_inf = all_res[1, :]
    num_total = num_inf + num_uninf
    prevs = (num_inf / num_total)

    form_data = pd.DataFrame({'date': dates,
                              'mean_load': mean_loads,
                              'prev': prevs,
                              'num_uninf': num_uninf,
                              'num_total': num_total,
                              'num_inf': num_inf})
    form_data = form_data.assign(month=lambda x: x['date'].dt.month,
                                 year=lambda x: x['date'].dt.year)

    # Set NAs to zero
    form_data.loc[form_data.mean_load.isnull(), "mean_load"] = 0
    form_data.loc[form_data.prev.isnull(), "prev"] = 0

    return(form_data)


def simple_temperature(vals, min_temp, max_temp, year_length=365):
    a = ((max_temp - min_temp) / 2)
    temps = a*(1 - np.cos(2*np.pi*vals / year_length)) + min_temp
    return(temps)


def get_seasonal_R0(base_params, tvals, abund, means,
                    min_temp, max_temp, comm,
                    time_step,
                    year_length=365):
    """
    Calculate seasonal R0/invasion criteria for species

    Parameters
    ----------
    base_params : dict
        The parameters defining the reduced dimension IPM
    tvals : array-like
        The time values over which a "season" is defined. Starts and ends at lowest temperature
    abund : array-like
        The same length as tvals. The host density at each time point
    means : array-like
        The same length a tvals. The host mean at each time point
    min_temp : float
        The minimum temperature over a season
    max_temp : float
        The maximum temperature over a season
    comm : Community object
        Contains functions for computing temperature-dependent effects.
    time_step : int
        The time step of the model
    seasonal_management : array-like
        Same length as tvals.  Reduce values by some percent at some time point.
    perturb_link: bool
        Determine whether or not the perturbation happens on the link scale or natural scale. If True,
        perturbation happens on the link scale.

    Returns
    -------
    : dict
        Keywords
            - R0: Seasonal, individual-based R0 using the time-invariant approach of Diekkmann.
                  This has the same biological interpretation as R0 traditionally does.
            - λ: The maximum eigenvalue of the Jacobian from the time-invariant seasonal matrix
            - alt_R0: An alternate (though equivalent) approach to calculating seasonal R0
    """

    # Holds seasonally-varying matrices
    Amats = []
    Fmats = []
    Mmats = []
    Mstars = []
    inpools = []

    # Specifies a seasonal cycle
    for i, t in enumerate(tvals):

        tp = base_params.copy()
        temp = simple_temperature(t, min_temp, max_temp, year_length=year_length) #simple_temperature(t, min_temp, max_temp, year_length=364)

        beta, a, mu0, lI, sigma_temp = comm.temperature_effects(tp, temp, 0, 0, 0)

        # Assuming no load-dependent loss of infection
        zsurv = comm.zsurv_fxn(temp)

        # Update if host is in the pool to transmit
        inpool = np.int(comm.in_pool(t / time_step)[0])  # Assuming only one species
        inpools.append(inpool)

        temperature_params = dict(a=a, sI=tp['sI'],
                                  sigma_full=tp['sigma_full'],
                                  b=tp['b'], lam=tp['lam'], inpool=inpool, zsurv=zsurv,
                                  s0=tp['s0'], lI=lI, mu0=mu0, mu_l=tp['mu_l'],
                                  sigma_l=tp['sigma_l'])

        Mstar = mstar_load_dependent_loss(temperature_params)
        Mstars.append(Mstar)

        # Approximate Mstar with the population-level mean.
        temperature_params['Mstar'] = means[i]

        # Fecundity matrix
        tF = buildF(dict(S=abund[i], beta=beta, s0=tp['s0'], inpool=inpool,
                         mu0=mu0))

        # Transition matrix
        tM = buildM_loss_dep(temperature_params)

        tA = tF + tM
        Amats.append(tA)
        Fmats.append(tF)
        Mmats.append(tM)

    # Compute the different invasion criteria: R0, lambda, and floquet
    fullJ = block_diag_offset(np.array(Amats))
    fullF = block_diag_offset(np.array(Fmats))
    fullM = block_diag_offset(np.array(Mmats))
    fullR = np.dot(fullF, np.linalg.inv(np.eye(fullM.shape[0]) - fullM))

    # Use the formula in Bacaer et al. 2012 for R0
    fullF_alt = block_diag(*Fmats)
    fullM_alt = block_diag(*[-1*tm for tm in Mmats])
    littleI = np.eye(*Mmats[0].shape)
    s = littleI.shape[0]

    num_mats = len(Mmats)
    for m in range(num_mats - 1):
        fullM_alt[m*s:(m*s + s), ((m*s) + s):((m*s + s) + s)] = littleI

    fullM_alt[-3:, :3] = littleI

    fullR_alt = np.dot(fullF_alt, np.linalg.inv(fullM_alt))

    # Floquet matrix
    # for i in range(len(Amats))[::-1]:
    #     if i == (len(Amats) - 1):
    #         Atot = Amats[i]
    #     else:
    #         Atot = Atot * Amats[i]

    return({"R0": np.max(np.abs(np.linalg.eigvals(fullR))),
            "λ": np.max(np.abs(np.linalg.eigvals(fullJ))),
            "alt_R0": np.max(np.abs(np.linalg.eigvals(fullR_alt)))})


if __name__ == '__main__':


    # Example of reduced model to explore

    params = yaml.safe_load(open("model_params/LA_params/spp_params_LISP.yml", 'r'))['params']

    # Play around with some different values
    params['a'] = 2
    params['a_temp'] = -0.5
    params['trans_beta'] = 1
    params['b'] = 0.5
    params['omega'] = np.log(500)
    params['K'] = 4
    params['sigma_full_temp'] = 0.2

    # Set-up model details
    model_params = {'time_step': 7, 'mean_temp': 15}

    spp_params = {}
    spp_params['spp1'] = params

    # Initial conditions
    init = {'spp1': np.array([0, 1.0, 1.0, 1.0]),
            'Z': np.array([1.0])}

    z_fxn = zsurv_fxn(model_params['time_step'])
    steps = 201
    start_date = pd.datetime(2016, 1, 1)

    # Get empirical temperature curve
    # temp_data, base_dat, longterm = mcmc.get_temp_data('LA')
    # temp_fxn, temp_fxn_cv, temp_fxn_mm = mcmc.build_temperature_fxn(temp_data, base_dat, longterm, 1,
    #                                                                 start_date=start_date)

    # Simple temperature function...assuming a constant temperature
    def temp_fxn_simple(x):
        return(14)

    temp_fxn = np.vectorize(temp_fxn_simple)

    comm = Community(spp_params, z_fxn, init, model_params, area=1.0,
                     init_temp=temp_fxn(0), load_dependent=False)

    res = comm.simulate(steps, temp_fxn=temp_fxn) # No temperature dependence

    # Plot results
    ps = 4
    time = np.arange(steps)*model_params['time_step']
    datetime = start_date + pd.to_timedelta(time, unit="D")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

    axes = axes.ravel()
    axes[0].plot(datetime, res[1, :] + res[2, :], '-', ms=ps, label="N")
    axes[0].plot(datetime, res[2, :], '-', ms=ps, label="I")
    axes[0].set_ylabel("Host density per $m^2$", size=18)

    axes[1].plot(datetime, res[3, :] / res[2, :], '-', ms=ps, label="mean")
    axes[1].set_ylabel("Mean log Bd load", size=18)
    axes[2].plot(datetime, res[4, :], '-', ms=ps, label="Z")
    axes[2].set_ylabel("Density per $m^2$", size=18)
    axes[2].set_xlabel("Date", size=18)
    axes[3].set_xlabel("Date", size=18)
    axes[3].plot(datetime, res[2, :] / (res[1, :] + res[2, :]), '-', ms=ps, label="I / (S + I)")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("Prevalence", size=18)
    for ax in axes:
        ax.legend()

    ax2 = axes[3].twinx()
    ax2.plot(datetime, temp_fxn(time), '-', color="red", alpha=0.3, label="Temperature, C")
    ax2.legend()
    ax2.set_ylabel("Temperature, C", size=18)

    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.show()
