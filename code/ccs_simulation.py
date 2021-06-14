import numpy as np
import pandas as pd
import yaml
import logging
import logging.handlers
import multiprocessing as mp
import scipy.interpolate as interpolate
import scipy.optimize as opt

# Configure logging file
logging.basicConfig(filename='ccs.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info("Initializing logger")

import full_model as full
import matplotlib.pyplot as plt
import importlib
import seaborn as sns
importlib.reload(full)

"""
For each DOD location (LA, TN, PA, VT), run a critical community size (CCS) analysis to see if
post-metamorphic leopard frogs can be responsible for long-term Bd persistence.
This entails stochastically simulating the best fit models for 10 years and
determining whether or not Bd persists over this time period.

This code can be used to reproduce the Figure 5 in the main manuscript
"""


def load_reduced_parameters(dod, spp, K, omega, model, loss_load, beta_prior):
    """
    Load the fitted parameters for the reduced IPM

    Parameters
    ----------
    dod : str
        DOD location
    spp : str
        Species at dod location
    K : float
        Assumed carrying capacity
    omega : float
        External zoospore contribution
    model : str
        Model name
    loss_load : str
        Either "" or "loss_load_", specifying whether loss of infection was
        intensity-dependent
    beta_prior : string
        Either "" or "beta_prior_" specifying whether or not to use the beta_prior
        fit.

    Returns
    -------
    : dict
        Dictionary with reduced parameter estimates

    """

    mcmc_res = pd.read_pickle("../results/pickled_results/{0}_{4}_{1}_allsites_{6}K={2}_{5}omega={3}.pkl".format(dod, spp, K, omega, model, loss_load, beta_prior))
    chains = mcmc_res['chains']
    adapt = mcmc_res['adapt_param']

    base_params = mcmc_res['base_params'][spp].copy()
    ndarray = np.array([mcmc_res['mcmc_results'][i]['params'][spp][:, 5*adapt:].T for i in range(chains)])

    med_params = np.percentile(ndarray, 50, axis=(0, 1))

    for i, p in enumerate(mcmc_res['parameters']):
        base_params[p] = med_params[i]

    # Potentially missing parameters
    hibernation_params = ['hibernation', 'hibernation_aquatic',
                          'hibernation_start', 'hibernation_end',
                          'sigma_full_temp']

    tparams = yaml.safe_load(open("model_params/{0}_params/spp_params_{1}.yml".format(dod, spp), "r"))
    for hp in hibernation_params:
        base_params[hp] = tparams['params'][hp]

    params_red = base_params.copy()
    params_red['base_temp'] = 15  # Baseline mean temperature for centering
    params_red['sigmaG'] = base_params['sigma_full']
    params_red['sigma0'] = base_params['sigma_full']

    return(params_red)


def get_full_ipm_params(params_red, constant_surv=True, constant_loss=True):
    """
    Build the parameters for the full IPM using the reduced IPM parameters

    Parameters
    ----------
    params_red : dict
        Parameters for the reduced IPM
    constant_surv : bool
        If True, assumes constant survival probability in the full IPM. False
        gives an IPM with load-dependent survival probability.
    constant_loss : bool
        If True, assumes constant loss of infection probability in the full IPM.
        False assumes loss of infection probability depends on pathogen load.

    Returns
    -------
    : dict
        Dictionary containing the parameters for a full IPM
    """

    # Dict
    params_full = {
        's0': params_red['s0'],
        'growth_fxn_inter': params_red['a'],
        'growth_fxn_temp': params_red['a_temp'],
        'growth_fxn_slope': params_red['b'],
        'growth_fxn_sigma': params_red['sigmaG'],
        'loss_fxn_inter': params_red['mu_l'],
        'loss_fxn_slope': params_red['sigma_l'],
        'init_inf_fxn_inter': params_red['mu0'],
        'init_inf_fxn_sigma': params_red['sigma0'],
        'surv_fxn_inter': params_red['mu_s'], #np.log(params_red['sI'] / (1 - params_red['sI'])),
        'surv_fxn_slope': params_red['sigma_s'],
        'trans_fxn_zpool': params_red['trans_beta'],
        'trans_fxn_temp': params_red['trans_beta_temp'],
        'max_load': 1000,
        'min_load': -1000,
        'shedding_prop': params_red['lam'],
        'fec': params_red['r'],
        'K': params_red['K'],
        'constant_surv': constant_surv,
        'constant_loss': constant_loss,
        'sI': params_red['sI'],
        'lI': 1 / (1 + np.exp(-params_red['lI'])),
        'larval_period': params_red['larval_period'],
        'aquatic': params_red['aquatic'],
        'breeding_start': params_red['breeding_start'],
        'breeding_end': params_red['breeding_end'],
        'hibernation': params_red['hibernation'],
        'hibernation_aquatic': params_red['hibernation_aquatic'],
        'hibernation_start': params_red['hibernation_start'],
        'hibernation_end': params_red['hibernation_end'],
        'base_temp': params_red['base_temp']
    }

    return(params_full)


def sim_full_ipm(mod, steps, stochastic=False):
    """
    Simulate the full IPM model

    Parameters
    ----------
    mod : FullModel
        Class for full IPMs defined in full_model.py
    steps : int
        Number of time steps over which to run the model
    stochastic : bool
        Whether or not the simulation should include demographic stochasticity

    Returns
    -------
    : tuple
        First item is an array of simulation results (state variables, time steps)
        Second item is an array of temperature values

    """

    sim = np.empty((len(mod.species[0].y) + 3, steps))
    temperature_vals = np.empty(steps)
    temperature_vals[0] = mod.temperature
    sim[:, 0] = mod.species[0].density_vect
    for step in range(1, steps):

        if stochastic:
            mod.update_stochastic()
        else:
            mod.update_deterministic()
        sim[:, step] = mod.species[0].density_vect
        temperature_vals[step] = mod.temperature

    return((sim, temperature_vals))


def summarize_sim(sim, mod):
    """
    Extract summary statistics from simulation

    Given a full IPM simulation, extract the summary statistics
    for easier processing

    Parameters
    ----------
    sim : array
        The first result in the tuple from sim_full_ipm
    mod : FullModel
        Class for full IPMs defined in full_model.py

    Returns
    -------
    : dict
        Summarized model output with the following keyword
        'tad': Tadpole density through time
        'z': zoospore density through time
        'n': Density of post-metamorphic hosts through time
        'n_uninf': Density of uninfected post-metamorphic through time
        'n_inf': Density of infected post-metamorphic through time
        'loads': All infected "classes"
        'loads_dists': Normalized infection classes
        'y': mid-points used in IPM approximation
        'mean': Mean of pathogen load distribution through time
        'var': Variance of pathogen load distribution through time
        'k': mean^2 / var

    """

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


def multiprocess_stochastic(processes, num_sims, params_full, init_dens_full,
                            dod, spp, ipm_params, area,
                            min_temp, max_temp, steps, logger):
    """ Multiprocess the stochastic simulations """

    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(sim_stochastic,
                                args=(i, params_full, init_dens_full, dod,
                                      spp, ipm_params, area,
                                      min_temp, max_temp, steps, logger))
               for i in np.arange(num_sims)]
    results = [p.get() for p in results]
    pool.close()

    # results = []
    # for i in range(num_sims):
    #     results.append(sim_stochastic(i, params_full, init_dens_stoch, dod,
    #                                   spp, ipm_params, area,
    #                                   min_temp, max_temp, steps))
    return(results)


def sim_stochastic(i, params_full, init_dens_full, dod, spp,
                   ipm_params, area, min_temp, max_temp, steps, logger):
    """

    Run an stochastic simulation of the model to test for long-term Bd persistence

    Parameters
    ----------
    i : int
        A integer for counting simulations
    params_full : dict
        Parameters for the full IPM
    init_dens : array-like
        Initial densities in each load class
    dod : str
        Dod/location name
    spp : str
        Species name
    ipm_params : dict
        Parameters that specify the details of the IPM discretization.
        min_size, max_size, bins, time_step
    area : float
        The area of of the aquatic "arena"
    min_temp: float
        Minimum temperature per season
    max_temp : float
        Maximum temperature per season
    steps : int
        Number of steps in the IPM simulation
    logger : Logger class
        For logging the progress of the simulation

    Returns
    -------
    : tuple
        (i, bool specifying whether there are either infected hosts or zoospores left in population after 10 years of simulation)
    """

    # Fit the
    logger.info("Simulation {0}".format(i))

    # Randomize starting condition
    N_inf = init_dens_full[2:-1].sum()
    inf_dist = np.random.multinomial(N_inf, init_dens_full[2:-1] / N_inf)
    init_dens_stoch = np.r_[np.ceil(init_dens_full[0:2]),
                            inf_dist,
                            np.ceil(init_dens_full[-1])]
    comm_params = {'time': 0,
                   'species': {spp: params_full},
                   'density': {spp: init_dens_stoch}}

    zsurv_fxn = full.zsurv_fxn_temperature(ipm_params['time_step'])
    full_mod = full.Community(dod, comm_params, ipm_params, zsurv_fxn,
                              area=area, base_temp=params_full['base_temp'],
                              init_temp=min_temp, min_temp=min_temp,
                              max_temp=max_temp)
    sim_full_stoch, temp_vals_stoch = sim_full_ipm(full_mod, steps,
                                                   stochastic=True)
    full_sim_stoch = summarize_sim(sim_full_stoch, full_mod)

    # 522 is approximately 10 years with 7 day time steps.
    prob_persist10yrs = (full_sim_stoch['n_inf'][522] > 0) or ((full_sim_stoch['z'][522] > 0))

    return((i, prob_persist10yrs))


if __name__ == '__main__':

    # (DOD/Location, species, (min, max temperature pairs), *model_indentifiers)
    dod_spp = [("PA", 'LIPI', (4, 25), "a_temp", "loss_load_", "ext_beta_prior_"),
               ("LA", 'LISP', (4, 30), "both_model", "loss_load_", "ext_beta_prior_"),
               ("TN", 'LISP', (4, 27), "a_temp", "loss_load_", "ext_beta_prior_")]

    # Basic full model parameters
    ipm_params = {
        'min_size': -20,
        'max_size': 25,
        'bins': 300,
        'time_step': 7
    }

    # Scales defines the area of the habitat
    scales = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] # [11000]
    steps = 530
    K = 4
    omega = 1 # This parameter is fit, but omega = 1 species a specific model
    density = 0.14 # A high density assumption 0.002
    num_sims = 300  # Number of stochastic simulations per scale
    processes = 4  # Number of cores
    year_length = 364
    increaseR0 = False

    for dod, spp, (min_temp, max_temp), model, loss_load, beta_prior in dod_spp:

        all_ccs = []

        for scale in scales:

            logger.info("Simulations for {1}, scale {0}".format(scale, dod))
            Sinit = density*scale
            area = 1*scale

            params_red = load_reduced_parameters(dod, spp, K, omega, model,
                                                 loss_load, beta_prior)

            # No external contribution.  This ensures that there is not
            # trivial Bd persistence.
            params_red['omega'] = -1000000

            # Adjust lambda to increase R0 if desired
            if increaseR0 and dod == "PA":
                params_red['lam'] = 4.5

            params_full = get_full_ipm_params(params_red,
                                              constant_surv=True,
                                              constant_loss=False)

            # Run a deterministic trajectory to get initial conditions.
            zsurv_fxn = full.zsurv_fxn_temperature(ipm_params['time_step'])
            init_dens_full = np.r_[0, Sinit, np.zeros(ipm_params['bins']), 10000]
            comm_params = {'time': 0,
                           'species': {spp: params_full},
                           'density': {spp: init_dens_full}}
            full_mod = full.Community(dod, comm_params, ipm_params, zsurv_fxn,
                                      area=area, base_temp=params_full['base_temp'],
                                      init_temp=min_temp, min_temp=min_temp,
                                      max_temp=max_temp, year_length=year_length)

            logger.info("Simulating deterministic model...")
            sim_full_det, temp_vals = sim_full_ipm(full_mod, steps, stochastic=False)
            full_sim_det = summarize_sim(sim_full_det, full_mod)
            logger.info("Simulation complete")

            # Set initial conditions
            init_dens_full = sim_full_det[:, -1]

            # Multiprocess stochastic simulations
            res = multiprocess_stochastic(processes, num_sims, params_full,
                                          init_dens_full,
                                          dod, spp, ipm_params, area,
                                          min_temp, max_temp, steps, logger)
            all_ccs.append(res)

        # Save results
        suffix = ""
        if increaseR0:
            suffix = suffix + "_increaseR0"
        pd.to_pickle({'results': all_ccs,
                      'abundances': density*np.array(scales),
                      'num_sims': num_sims,
                      'model': model,
                      'loss_load': loss_load,
                      'K': K,
                      'ipm_params': ipm_params}, "../results/pickled_results/{0}_ccs_results{1}.pkl".format(dod, suffix))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    dod_sites = ['LA', 'TN', 'PA']

    # Loop through sites and plot CCS
    for i, dod in enumerate(dod_sites):

        tres2 = pd.read_pickle("../results/pickled_results/{0}_ccs_results.pkl".format(dod))

        tprobs = np.array([np.mean(list(zip(*x))[1]) for x in tres2['results']])

        if dod == "PA":
            lab = dod + r", $R_0 = 1.5$"
        else:
            lab = dod
        ax.plot(tres2['abundances'], tprobs, '--', color=sns.color_palette()[i], label=lab)

        if dod == 'PA':
            tres_highR0 = pd.read_pickle("../results/pickled_results/{0}_ccs_results_increaseR0.pkl".format(dod))
            tprobs2 = np.array([np.mean(list(zip(*x))[1]) for x in tres_highR0['results']])
            ax.plot(tres_highR0['abundances'], tprobs, '-', lw=3, alpha=0.5, color=sns.color_palette()[i], label=dod + r", $R_0 = 20$")

        # Plot CCS point
        if dod in ['LA', 'TN']:
            interp_fxn = interpolate.interp1d(tres2['abundances'], tprobs)
            ccs = opt.brentq(lambda x: interp_fxn(x) - .5, 15, 139)
            ax.plot([ccs], [0.5], 'o', color='black')
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            if dod == 'TN':
                ax.hlines(0.5, 14, ccs, color='black', linestyle=':', linewidth=1)
            ax.vlines(ccs, 0, .5, color='black', linestyle=':', linewidth=1)

    ax.legend(loc="center right", prop={'size': 9}, frameon=False)
    ax.set_ylabel("Prob. Bd persists >10 yr")
    ax.set_xlabel("Maximum leopard frog seasonal abundance")
    fig.savefig("../results/plots/ccs_persistence.pdf", bbox_inches="tight")



















