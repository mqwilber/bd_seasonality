import numpy as np
import scipy.stats as stats
import reduced_model as red
from scipy.optimize import fmin

""" Full host-parasite IPM w/ tadpole stage """


class Community:
    """
    Community class with host species.  Allows for the inclusion of multiple
    host species in the model.
    """

    def __init__(self, name, comm_params, ipm_params, zsurv_fxn, area=1,
                 base_temp=15, init_temp=0, min_temp=4.0, max_temp=30.0,
                 year_length=364):
        """
        Parameters
        ----------
        name : str
            Name of the model
        comm_params :
            'time': int, the starting time. 1 is a good default.
            'species': dict, keywords are species names and each keyword
                       looks up a dictionary with species specific parameters
                       needed for the Species class.
            'density': dict, keywords are species and each looks up a vector
                       of initial densities for that species.
        ipm_params : dict
            'min_size': The lower bound of the IPM integral
            'max_size': The upper bound of the IPM integral
            'bins': Number of discretized classes in the IPM integral
            'time_step': The number of days in a time step
        area : float
            The area of the arena. Default is one which just models density.
        base_temp : float
            A temperature in Celsius that is used for centering temperature
            variable
        init_temp : float
            Initial temperature at the start of the simulation.
        min_temp: float
            Minimum yearly temperature
        max_temp: float
            Maximum yearly temperature
        year_length : int
            Define the length of a year

        """

        self.species = []
        for name, spp_params in comm_params['species'].items():
            self.species.append(Species(name, spp_params, ipm_params,
                                        comm_params['density'][name], area,
                                        year_length))

        self.time = comm_params['time']
        self.update_zoospore_pool()
        self.ipm_params = ipm_params
        self.base_temp = base_temp  # For centering temperature
        self.temperature = init_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.zsurv_fxn = zsurv_fxn
        self.zsurv = self.zsurv_fxn(self.temperature)
        self.year_length = year_length

    def update_zoospore_pool(self):
        """ Calculate the zoospore pool """
        self.zpool = np.sum([spp.density_vect[-1] for spp in self.species])

    def __str__(self):
        """ String representation of community object """
        return("\n".join([spp.__str__() for spp in self.species]))

    def update_time(self):
        self.time = self.time + self.ipm_params['time_step']

    def update_zsurv(self):
        self.zsurv = self.zsurv_fxn(self.temperature)

    def update_temperature(self):

        self.temperature = simple_temperature(self.time, self.min_temp,
                                              self.max_temp,
                                              year_length=self.year_length)

    def update_deterministic(self):
        """ Update all species one time-step, deterministically """

        [spp.update_deterministic(self.zpool, self.time, self.temperature, self.zsurv)
            for spp in self.species]

        self.update_zoospore_pool()
        self.update_time()
        self.update_temperature()
        self.update_zsurv()

    def update_stochastic(self):
        """ Update all species one time-step, stochastically """

        [spp.update_stochastic(self.zpool, self.time, self.temperature, self.zsurv)
            for spp in self.species]

        self.update_zoospore_pool()
        self.update_time()
        self.update_temperature()
        self.update_zsurv()

    def simulate(self, steps, stochastic=False):
        """
        Simulate the community for 'steps' time steps either stochastically or
        deterministically

        Returns
        -------
        : dict
            Key word for each species in the community looking on S x steps + 1
            matrix with density vector at each time step.
        """

        ts = self.ipm_params['time_step']
        time_vals = np.arange(0, steps*ts + ts, step=ts)

        # Initial arrays to hold results
        species_res = {}
        for spp in self.species:

            all_res = np.empty((spp.ipm_params['bins'] + 2, steps + 1))
            all_res[:, 0] = np.copy(spp.density_vect)  # Initialize
            species_res[spp.name] = all_res

        if stochastic:
            for i in range(steps):
                self.update_stochastic()

                for spp in self.species:
                    species_res[spp.name][:, i + 1] = spp.density_vect
        else:
            for i in range(steps):
                self.update_deterministic()

                for spp in self.species:
                    species_res[spp.name][:, i + 1] = spp.density_vect

        return((time_vals, species_res))


class Species:
    """
    Host species object that builds an IPM for a particular host species
    """

    def __init__(self, name, spp_params, ipm_params, init_density_vect,
                 area=1.0, year_length=364):
        """
        Parameters
        ----------

        name : str
            Species name
        spp_params : dict
            'growth_fxn_*':
                - 'inter': The intercept of the growth function
                - 'slope': The effect of previous Bd load on current Bd load
                - 'sigma': The standard deviation of the Bd growth function
            'loss_fxn_*':
                - 'inter': The intercept of the Bd loss probability function
                - 'slope': The effect of current Bd load on log odds loss
            'init_inf_fxn_*':
                - 'inter': The intercept of the initial infection function
                - 'sigma': The standard deviation in the initial infection load
            'surv_fxn_*': dict('inter', 'slope', 'temp')
                - 'inter': The LD50 of the survival function
                - 'slope': Inverse measure of the steepness of the survival function
            'constant_surv': If True, infected survival probability is constant
                             with probability s_I
            'constant_loss': If True, loss of infection is constant with probability
                             lI
            'shedding_prop': Proportionality constant for zoospore shedding from
                             infected individual.
            'trans_fxn_*':
                - 'zpool': The transmission coefficient between zoospore pool
                           density and infection probability in a time step.
            'nu': Zoospore survival probability

        ipm_params : dict
            'min_size': The lower bound of the IPM integral
            'max_size': The upper bound of the IPM integral
            'bins': Number of discretized classes in the IPM integral
        area : float
            The area of the arena. Default is one which just models density.
        year_length : int
            Define the length of a year

        """

        self.name = name
        self.spp_params = spp_params  # Holds parameters related to species biology
        self.ipm_params = ipm_params  # Holds parameters related to IPM implementation
        self.density_vect = init_density_vect
        self.area = area
        self.year_length = year_length


        # Derived parameters
        self.spp_params['repro_time'] = red.get_repro_time(self.spp_params['breeding_start'],
                                                           self.spp_params['breeding_end'])

        t = (self.spp_params['larval_period'] / self.ipm_params['time_step'])
        self.spp_params['meta_prob'] = (1.0 / t)

        # Assume only 20% survival period over average larval period
        self.spp_params['s_tadpole'] = (0.2)**(1.0 / t)

        # Make ipm parameters
        ipm_params = set_discretized_values(ipm_params['min_size'],
                                            ipm_params['max_size'],
                                            ipm_params['bins'])

        self.y = ipm_params['y']  # midpoints
        self.h = ipm_params['h']  # Width of interval
        self.bnd = ipm_params['bnd']
        self.matrix_error = False

        # Account for breeding spanning years
        self.bstart = np.array([self.spp_params['breeding_start']])
        self.bend = np.array([self.spp_params['breeding_end']])
        self.aquatic = np.array([self.spp_params['aquatic']]).astype(np.bool)
        self.wrap = self.bstart > self.bend
        new_starts = self.bend[self.wrap]
        new_ends = self.bstart[self.wrap]
        self.bstart[self.wrap] = new_starts
        self.bend[self.wrap] = new_ends

        # Account for hibernation spanning years
        self.hstart = np.array([self.spp_params['hibernation_start']])
        self.hend = np.array([self.spp_params['hibernation_end']])
        self.hibernation = np.array([self.spp_params['hibernation']]).astype(np.bool)
        self.hibernation_aquatic = np.array([self.spp_params['hibernation_aquatic']]).astype(np.bool)
        self.hwrap = self.hstart > self.hend
        new_hstarts = self.hend[self.hwrap]
        new_hends = self.hstart[self.hwrap]
        self.hstart[self.hwrap] = new_hstarts
        self.hend[self.hwrap] = new_hends

    def build_ipm_matrix(self, temperature):
        """
        Function to build the IPM portion of the host-parasite IPM
        """
        #
        X, Y = np.meshgrid(self.y, self.y)
        _, Y_upper = np.meshgrid(self.y, self.bnd[1:])
        _, Y_lower = np.meshgrid(self.y, self.bnd[:-1])
        G = _growth_fxn(Y_lower, Y_upper, X, self.spp_params, temperature)
        S = _survival_fxn(self.y, self.spp_params)
        L = _loss_fxn(self.y, self.spp_params)

        P = np.dot(G, np.diagflat(S*(1 - L)))

        # All column sums should be less than 1
        if np.any(P.sum(axis=0) > 1):
            self.matrix_error = True

        # Save the intermediate kernels results
        self.S = S
        self.L = L
        self.G = G
        self.P = P

    def build_full_matrix(self, zpool, time, temperature, zsurv):
        """
        The full transition matrix that includes tadpoles, susceptible hosts
        and the zoospore pool. Used to update the model one-time step within a
        season.

        Parameters
        ----------
        zpool: float
            The density of zoospores in the pool
        time: int
            Time (on daily scale from start)
        temperature : float
            The current temperature
        zsurv : float
            Zoospore survival probability in a time step
        """

        centered_temp = temperature - self.spp_params['base_temp']
        self.build_ipm_matrix(centered_temp)
        inpool = self.in_pool(time).astype(np.int)[0]

        inf_prob = _trans_fxn(zpool, self.spp_params, centered_temp, self.area)*inpool

        y_lower = self.bnd[:-1]
        y_upper = self.bnd[1:]
        init_inf = _init_inf_fxn(y_lower, y_upper,
                                 self.spp_params,
                                 centered_temp)

        # Susceptible -> infected
        colS = np.r_[0,
                     self.spp_params['s0']*(1 - inf_prob),
                     self.spp_params['s0']*inf_prob*init_inf]

        # Infected -> susceptible

        # Density-dependent recruitment
        N = np.sum(self.density_vect[1:-1])
        dd_recruit = np.exp(-(np.exp(self.spp_params['K']) / self.area)*N)

        rowS = np.r_[self.spp_params['s_tadpole']*self.spp_params['meta_prob']*dd_recruit,
                     self.spp_params['s0']*(1 - inf_prob),
                     self.S*self.L]

        rowminus1 = np.r_[0, 0, self.spp_params['shedding_prop']*np.exp(self.y)*inpool, zsurv]

        # Add Tadpoles, Susceptible, zoospore vectors to matrix
        T = np.zeros(np.array(self.P.shape) + 3)
        T[0, 0] = self.spp_params['s_tadpole']*(1 - self.spp_params['meta_prob'])
        T[:-1, 1] = colS
        T[1, :-1] = rowS
        T[2:-1, 2:-1] = self.P
        T[-1, :] = rowminus1
        T[-1, -1] = zsurv

        self.T = T

        # Add seasonal reproduction.
        self.F = np.zeros(T.shape)
        year_time = (time % self.year_length)
        lower = year_time - self.ipm_params['time_step']
        if ((self.spp_params['repro_time'] > lower) &
           (self.spp_params['repro_time'] <= year_time)):

            repro = self.spp_params['fec']
            full_repro = np.r_[0, np.repeat(repro, self.P.shape[0] + 1), 0]
            self.F[0, :] = full_repro

    def in_pool(self, time):
        """
        For a given time of year, check whether species is in the pool.

        Accounts for hibernation and breeding.

        Parameters
        ----------
        time : int
            Time in days

        Returns
        -------
        : Array of bools specifying whether or not species is in
          pool at time t.

        """

        year_time = (time % self.year_length)
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

    def model_R0(self, Sinit, temperature, time):
        """
        Calculate R0 for the full IPM model

        Parameters
        ----------
        Sinit : float
            Number of susceptible individuals at disease-free equilibrium

        Returns
        -------
        : tuple
            (next generation matrix, R0)
        """

        # U matrix

        # TODO, Incorporate temp dependence.
        self.build_ipm_matrix(temperature)
        inpool = self.in_pool(time).astype(np.int)[0]
        Ured = np.copy(self.P)

        row1 = np.exp(self.y)*self.spp_params['shedding_prop']*inpool

        U1 = np.vstack((Ured, row1))
        col1 = np.r_[np.zeros(len(row1)), self.spp_params['nu']][:, np.newaxis]
        U = np.hstack((U1, col1))

        # Build reproduction matrix
        F = np.zeros(U.shape)
        y_lower = self.bnd[:-1]
        y_upper = self.bnd[1:]
        init_inf = _init_inf_fxn(y_lower, y_upper, self.spp_params)
        init_col = Sinit*self.spp_params['s0']*self.spp_params['trans_fxn_zpool']*init_inf
        F[:-1, -1] = init_col

        minusUinv = np.linalg.inv(np.eye(len(U)) - U)
        Rmat = np.dot(F, minusUinv)

        return((Rmat, np.max(np.linalg.eigvals(Rmat))))

    def update_deterministic(self, zpool, time, temperature, zsurv):
        """
        One time-step update, deterministic
        """

        self.build_full_matrix(zpool, time, temperature, zsurv)
        self.density_vect = np.dot(self.F + self.T, self.density_vect)

    def update_stochastic(self, zpool, time, temperature, zsurv):
        """
        One time-step update, stochastic
        """

        # Create
        self.build_full_matrix(zpool, time, temperature, zsurv)

        T = self.T.copy()
        T[-1, :-1] = 0  # Remove zoospore fertility but keep survival
        death_probs = 1 - T.sum(axis=0)
        death_probs[death_probs < 0] = 0
        Taug = np.vstack([T, death_probs])  # Add death class
        Taug = Taug / Taug.sum(axis=0)

        # Production of zoospores. Using the fact that a sum of Poissons
        # is Poisson.
        Zrepro = self.T[-1, 2:-1]
        inf_hosts = np.ceil(self.density_vect[2:-1]).astype(np.int)
        gained_z = stats.poisson(np.sum(Zrepro*inf_hosts)).rvs(size=1)

        # Moving to new infection, non-infected classes
        updated = np.array([np.random.multinomial(n, p) for n, p in
                            zip(self.density_vect.astype(np.int), Taug.T)])
        new_density = updated.sum(axis=0)[:-1]  # Don't track dead individuals
        new_density[-1] = new_density[-1] + gained_z

        # Add new susceptible individuals from reproduction
        n = self.density_vect[1:-1]
        repro = stats.poisson(np.sum(self.F[0, 1:-1]*n)).rvs(size=1)
        new_density[0] = new_density[0] + repro

        self.density_vect = new_density

    def temperature_effects(self, params, temp):
        """
        Update parameters with temperature effects

        Parameters
        ----------
        params : dict
            Species specific parameters
        temp : float
            Temperature

        Returns
        -------
        : tuple
            (beta (transmission), a (pathogen growth),
             mu0 (initial infection))
        """

        # Temperature effects
        base_temp = temp - self.model_params['mean_temp']

        # Transmission
        linbeta = (base_temp*params['trans_beta_temp'])
        beta = (params['trans_beta'] / self.area *
                np.exp(linbeta)) # m^2 / 7 days = 1 time step

        # Growth
        a = (params['a'] +
             params['a_temp']*base_temp)
        mu0 = a  # We assume mu0 and a are identical

        return((beta, a, mu0))

    def __str__(self):
        return("Species name: {0}\nCurrent density: {1}".format(
               self.name, np.sum(self.density_vect[:-1])))


def _growth_fxn(x_next_lower, x_next_upper, x_now, params, temperature):
    """ The Bd growth function """

    max_load = params['max_load']
    min_load = params['min_load']
    x_now[x_now > max_load] = max_load
    x_now[x_now < min_load] = min_load

    μ = (params['growth_fxn_inter'] + params['growth_fxn_temp']*temperature
         + params['growth_fxn_slope']*x_now)
    σ = params['growth_fxn_sigma']
    norm_fxn = stats.norm(loc=μ, scale=σ)
    prob = norm_fxn.cdf(x_next_upper) - norm_fxn.cdf(x_next_lower)

    return(prob)


def _survival_fxn(x_now, params):
    """ The host survival function """

    if not params['constant_surv']:
        u = params['surv_fxn_inter']
        prob = stats.norm(loc=u, scale=params['surv_fxn_slope']).sf(x_now)
    else:
        prob = np.repeat(params['sI'], len(x_now))

    return(prob*params['s0'])


def _loss_fxn(x_now, params):
    """ The loss of infection function """

    if not params['constant_loss']:
        u = params['loss_fxn_inter']
        prob = stats.norm(loc=u, scale=params['loss_fxn_slope']).sf(x_now)
    else:
        prob = np.repeat(params['lI'], len(x_now))

    return(prob)


def _trans_fxn(zpool, params, temperature, area):
    """ The transmission function """

    beta = (params['trans_fxn_zpool'] / area)*np.exp(params['trans_fxn_temp']*temperature)
    return(1 - np.exp(-(beta*zpool)))


def _init_inf_fxn(x_next_lower, x_next_upper, params, temperature):
    """ The initial infection function """

    μ = params['growth_fxn_inter'] + params['growth_fxn_temp']*temperature
    σ = params['growth_fxn_sigma']
    norm_fxn = stats.norm(loc=μ, scale=σ)
    prob = norm_fxn.cdf(x_next_upper) - norm_fxn.cdf(x_next_lower)

    return(prob)


def set_discretized_values(min_size, max_size, bins):
    """
    Calculates the necessary parameters to use the midpoint rule to evaluate
    the IPM model

    Parameters
    ----------
    min_size : The lower bound of the integral
    max_size : The upper bound of the integral
    bins : The number of bins in the discretized matrix

    Returns
    -------
    : dict
        min_size, max_size, bins, bnd (edges of discretized kernel), y (midpoints),
        h (width of cells)
    """

    # Set the edges of the discretized kernel
    bnd = min_size + np.arange(bins + 1)*(max_size-min_size) / bins

    # Set the midpoints of the discretizing kernel. Using midpoint rule for evaluation
    y = 0.5 * (bnd[:bins] + bnd[1:(bins + 1)])

    # Width of cells
    h = y[2] - y[1]

    return(dict(min_size=min_size,
                max_size=max_size,
                bins=bins, bnd=bnd, y=y,
                h=h))


def simple_temperature(time, min_temp, max_temp, year_length=364):
    """ Simple sinusoidal temperature fxn """

    a = ((max_temp - min_temp) / 2)
    temperature = a*(1 - np.cos(2*np.pi*time / year_length)) + min_temp
    return(temperature)


def zsurv_fxn_temperature(time_step):
    """
    Temperature-dependent zoospore survival

    Parameters
    ----------
    time_step: int
        Time-step in days of the IPM model

    Returns
    -------
    : Interpolated temperature-dependent zoospore survival function from
      Woodhams et al. 2008 (Ecology)

    """

    # Observed Zoospore survival from Woodhams et al. 2008
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

    # This is a faster function than interpolation
    def intp(x):
        return((1 / (1 + np.exp((x - ld50) / s)))*maxsurv)

    # intp = interp1d(temp_vals, obs_surv, kind="linear", bounds_error=False,
    #                 fill_value=(obs_surv[0], obs_surv[-1]))

    return(intp)


def sim_full_ipm(mod, steps, stochastic=False):
    """ Simulate the full IPM model """

    sim = np.empty((len(mod.species[0].y) + 3, steps))
    sim[:, 0] = mod.species[0].density_vect
    for step in range(1, steps):

        if stochastic:
            mod.update_stochastic()
        else:
            mod.update_deterministic()
        sim[:, step] = mod.species[0].density_vect

    return(sim)


def summarize_sim(sim, mod):
    """ Extract summary statistics from simulation """

    loads = sim[2:-1, :]
    load_dists = loads / loads.sum(axis=0)
    y = mod.species[0].y
    mean_loads = (load_dists * y[:, np.newaxis]).sum(axis=0)
    var_loads = (load_dists * y[:, np.newaxis]**2).sum(axis=0) - mean_loads**2
    k = mean_loads**2 / var_loads
    abund_total = sim[1:-1, :].sum(axis=0)
    abund_uninf = sim[1, :]
    tads = sim[0, :]
    z = sim[-1, :]

    sum_dict = {
        'tad': tads,
        'z': z,
        'n': abund_total,
        'n_uninf': abund_uninf,
        'n_inf': abund_total - abund_uninf,
        'loads': loads,
        'load_dists': load_dists,
        'y': y,
        'mean': mean_loads,
        'var': var_loads,
        'k': k
    }
    return(sum_dict)

