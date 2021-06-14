import os
os.environ["OMP_NUM_THREADS"] = "1" # Have numpy only use one thread to play nicely with multiprocessing
import pandas as pd
import reduced_model as comm
import abc_sampler as abc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.interpolate as interp
import yaml
import scipy.stats as stats
import logging
import logging.handlers
import importlib
import arviz as az
import multiprocessing as mp
from copy import deepcopy
importlib.reload(comm)
importlib.reload(abc)

"""
Script for fitting reduced IPMs to field data.  The full approach is described
in the supplementary material.

"""

# Configure logging file
logging.basicConfig(filename='mcmc.log', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set up file paths
home_path = os.path.expanduser("~")
data_dir = os.path.join("..", "data")
base_temp_path = os.path.join(data_dir,
                              "temperature_data",
                              "historical_temperature_data")
temperature_path = os.path.join(data_dir,
                                "temperature_data",
                                "measured_temperature_data")


def load_and_format_data(datapath, exclude=None):
    """
    Load and format the field data.  This function also excludes all frogs that
    were sampled haphazardly and not on specified survey dates.

    Parameters
    ----------
    datapath : str
        Path to the SERDP data
    exclude : list of tuple or None
        If tuple, in format (DOD_location, pd.Datetime, sites). Drops all samples for
        the given DOD location before the specified datetime.

    Return
    ------
    : DataFrame

    """

    fulldat = pd.read_csv(datapath)

    fulldat = (fulldat
               .assign(form_date=lambda x: pd.to_datetime(x.Date))
               .assign(year=lambda x: [d.year for d in x.form_date],
                       month=lambda x: [d.month for d in x.form_date],
                       Bd_pos=lambda x: ((x.Sample_Bd == 'positive')
                                         .astype(np.int)),
                       Bd_load=lambda x: np.where(
                                         np.isnan(x.Sample_Bd_swab_qty.values),
                                         0, x.Sample_Bd_swab_qty.values)))

    # Drop all amphibians that aren't adult or juvenile
    lifestage = ['Adult', "Juvenile"]
    fulldat = fulldat[fulldat.Life_stage.isin(lifestage)]

    if exclude is not None:
        tdat = fulldat.copy(deep=True)
        for dod, date, sites in exclude:
            tdat = tdat[~((tdat.DOD_location == dod) & (tdat.form_date < date) & (tdat.Site_code.isin(sites)))]

        fulldat = tdat

    return(fulldat)


def build_temperature_fxn(temp_data, base_dat, longterm, sitenum,
                          start_date=pd.datetime(2010, 1, 1),
                          use_hobo=False):
    """
    Build three temperature fxns.

    1. One that returns mean temperature
    2. One that returns the coefficient of variation in temperature
    3. One the returns the temperature mismatch from 20 year average.

    Parameters
    ----------
    temp_data : DataFrame
        HOBO data temperature at the site-level
    base_dat : DataFrame
        Historical temperature from gridMET
    sitenum : float
        The site number within the DOD location
    start_date : pd.datetime
        The first date to include with the NOAA data.

    Returns
    -------
    : tuple
        (mean temp fxn, cv temp fxn, mismatch fxn)
    """
    # Select particular site
    longterm_temp = longterm[longterm.site_number == sitenum].reset_index(drop=True)
    longterm_temp = longterm_temp[['month', 'day', 'datetime', 'mean_temperature']].sort_values(by='datetime').reset_index(drop=True)
    meanlong = longterm_temp.set_index("datetime").rolling(7, center=True, min_periods=3).mean().mean_temperature.values
    longterm_temp.loc[:, 'mean_temperature'] = meanlong
    base_dat_temp = base_dat[base_dat.site_number == sitenum]

    # Use the hobo data
    if use_hobo:
        site_temp = temp_data[temp_data.site == sitenum].sort_values(by="datetime").reset_index(drop=True)
        site_temp.loc[:, "datetime"] = pd.to_datetime(site_temp.datetime)
        site_temp = site_temp.assign(datetime=lambda x: pd.to_datetime(x.datetime.dt.date))
        site_temp = site_temp[~site_temp.water_temperature_C.isna()].reset_index(drop=True)
        # Get mean daily temperature
        site_temp = (site_temp.groupby("datetime")
                              .agg({'water_temperature_C': np.mean}).reset_index())
    else:
        site_temp = base_dat_temp[base_dat_temp.datetime >= pd.datetime(2017, 1, 1)]
        site_temp = site_temp.rename(columns={'mean_temperature':
                                              'water_temperature_C'})
        site_temp = (site_temp.groupby("datetime")
                              .agg({'water_temperature_C': np.mean}).reset_index())

    # For the given date, get the variability in temperature over seven days,
    # centered at the current value.  7 is the model time step. Still calculate
    # the variance at the edges.
    mean_temp = site_temp.set_index("datetime").rolling(7, center=True, min_periods=3).mean().reset_index()
    var_temp = site_temp.set_index("datetime").rolling(7, center=True, min_periods=3).var().reset_index()

    # Get time deltas for hobo data
    timedelta = (mean_temp.datetime.iloc[1:].reset_index(drop=True) -
                 mean_temp.datetime.iloc[:-1].reset_index(drop=True))
    time_seconds = timedelta.dt.total_seconds().values

    tempC_all = base_dat_temp.set_index("datetime").mean_temperature
    tempC = tempC_all.rolling(7, center=True, min_periods=3).mean().reset_index().mean_temperature.values
    varC = tempC_all.rolling(7, center=True, min_periods=3).var().reset_index().mean_temperature.values
    base_dat_temp = base_dat_temp.assign(pred_temp=tempC, pred_var=varC)
    start_obs = mean_temp.datetime.min()

    start_noaa = start_date
    base_trun = base_dat_temp[(base_dat_temp.datetime >= start_noaa) &
                              (base_dat_temp.datetime < start_obs)]
    noaa_end = base_trun.datetime.max()
    noaa_seconds = (base_trun.datetime[1:].reset_index(drop=True) -
                    base_trun.datetime[:-1].reset_index(drop=True)).dt.total_seconds().values

    # Combine the deltas
    diff = (start_obs - noaa_end).total_seconds()
    alldeltas = np.r_[0, noaa_seconds,
                      diff, time_seconds]

    alltemps = np.r_[base_trun.pred_temp.values, mean_temp.water_temperature_C]
    alltemps_var = np.r_[base_trun.pred_var.values, var_temp.water_temperature_C]
    allseconds = np.cumsum(alldeltas)
    alldays = allseconds / (60*60*24)

    # Get temperature mismatch
    fulldf = (pd.DataFrame({'datetime': start_date + pd.to_timedelta(alldays, unit="D"),
                           'mean_temperature': alltemps})
                .assign(month=lambda x: x.datetime.dt.month,
                        day=lambda x: x.datetime.dt.day))
    ind = ['month', 'day']
    mismatch_df = (fulldf.set_index(ind).join(longterm_temp.set_index(ind), rsuffix="_lt")
                         .reset_index().sort_values(by="datetime"))
    mismatch = (mismatch_df.mean_temperature - mismatch_df.mean_temperature_lt).values

    # Interpolate fxn
    interp_temp = interp.interp1d(alldays, alltemps)
    interp_temp_cv = interp.interp1d(alldays, np.sqrt(alltemps_var) / np.abs(alltemps))
    interp_temp_mm = interp.interp1d(alldays, np.abs(mismatch))

    # Function for temperature mean and temperature variability
    def temp_fxn(time):
        day = time % np.max(alldays)
        temp = interp_temp(day)
        return(temp)

    def temp_fxn_cv(time):
        day = time % np.max(alldays)
        cv_temp = interp_temp_cv(day)
        return(cv_temp)

    def temp_fxn_lt(time):
        day = time % np.max(alldays)
        cv_temp = interp_temp_mm(day)
        return(cv_temp)

    return((temp_fxn, temp_fxn_cv, temp_fxn_lt))


def plot_obs_pred(obs_dat, all_sim_dat=None, alpha=0.5):
    """
    Plot the observed and predicted trajectories to compare fits

    Parameters
    ----------
    obs_data : DataFrame
        Observed Bd field data
    all_sim_data : list-like
        Each entry contains simulated data

    Returns
    -------
    : matplotlib axes
    """

    fig, axes = plt.subplots(2, 2, figsize=(10, 4), sharex=True)
    axes = axes.ravel()

    try:
        _ = obs_dat.date
    except AttributeError:
        obs_dat = obs_dat.assign(date=lambda x: [pd.datetime(yr, m, 1) for yr, m in
                                                 zip(x.year, x.month)])

    try:
        axes[0].plot(obs_dat.date, obs_dat.num_total, '-o', lw=1, zorder=3, ms=3)
    except AttributeError:
        print("Observed density data not found")
    axes[1].plot(obs_dat.date, obs_dat.prev, '-o', lw=1, zorder=3, ms=3)
    axes[2].plot(obs_dat.date, obs_dat.mean_load, '-o', lw=1, zorder=3, ms=3)

    for ax in axes:
        ax.tick_params(axis='x', rotation=90)

    if all_sim_dat:

        for sim_dat in all_sim_dat:

            try:
                _ = sim_dat.date
            except AttributeError:
                sim_dat = sim_dat.assign(date=lambda x: [pd.datetime(yr, m, 1) for
                                         yr, m in zip(x.year, x.month)])

            try:
                axes[0].plot(sim_dat.date, sim_dat.num_total, '-',
                             alpha=alpha, color=sns.color_palette()[1])
            except AttributeError:
                print("Simulated density data not found")
            axes[1].plot(sim_dat.date, sim_dat.prev, '-', alpha=alpha)
            axes[2].plot(sim_dat.date, sim_dat.mean_load, '-', alpha=alpha)

    #axes[0].set_ylim(0, 60)
    axes[1].set_ylim(-0.05, 1.1)
    #axes[2].set_ylim(-2, 10)
    axes[0].set_title("Host density")
    axes[1].set_title("Prevalence")
    axes[2].set_title("Mean load")
    axes[3].tick_params(bottom=False, left=False,
                        labelbottom=False, labelleft=False)
    for pos in ['right', 'left', 'bottom', 'top']:
        axes[3].spines[pos].set_visible(False)

    ylabs = ['Density', 'Prev', 'ln Bd load', '']
    for i, ax in enumerate(axes):
        ax.set_ylabel(ylabs[i])

    return(axes)


def logpost(obsdat, countdat, simdat, params, param_priors, area, mean_temp=15):
    """
    Compute the log posterior distribution up to a normalizing constant.
    Accounts for detection probability of Bd loads.

    Parameters
    ----------
    obsdat : DataFrame
        Observed Bd load and prevalence data on the individual level
    countdat : DataFrame
        Observed abundance for species at sample dates
    simdat : DataFrame
        Simulated Bd load and prevalence data (mean and prevalence)
    params : array-like
        Current parameter estimates at which to compute likelihood
    param_priors : ABC object
        Contains information on priors and Jacobians.
    area : float
        Site area for scaling density

    Returns
    -------
    : tuple
        (posterior log-likelihood up to a normalizing constant,
         log-likelihood up to a normalizing constant)

    """

    # From DiRenzo et al. 2018, MEE. Detection probability parameters
    # To avoid having to estimate these

    # Pathogen extinction occurred. Parameters are impossible
    if simdat.prev.isna().any() or simdat.mean_load.isna().any():
        return((-np.inf, -np.inf))

    # Chosen to match Figure 1 qPCR curve from DiRenzo et al. 2018, MEE
    alpha = 0.5
    beta = 0.7 / np.log(10) # We are working on the natural log scale
    shift = alpha / -beta

    if len(obsdat) != 0:

        data = pd.merge_asof(obsdat, simdat, direction="nearest", on='date')

        # Count data
        # cdata = pd.merge_asof(countdat, simdat, direction="nearest", on='date')

        pos = data[data.Bd_load > 0]
        neg = data[data.Bd_load == 0]

        # Allow sigma_full to vary with temperature
        sigma = params[np.where(np.array(param_priors.params) == 'sigma_full')[0][0]]
        if 'sigma_full_temp' in param_priors.params:
            sigma_temp = params[np.where(np.array(param_priors.params) == 'sigma_full_temp')[0][0]]
        else:
            sigma_temp = 0
        sigma_full_temp = sigma*np.exp(sigma_temp * (data.temperature - mean_temp))
        sigma_full_temp_pos = sigma_full_temp[data.Bd_load > 0].values
        sigma_full_temp_neg = sigma_full_temp[data.Bd_load == 0].values

        # Detection probability given load
        detect_prob_pos = 1 / (1 + np.exp(-(alpha + beta*np.log(pos.Bd_load.values))))

        # Failure to detect probability when load is unknown.
        nodetect_prob_neg = 1 - logistic_normal(neg.mean_load - shift, sigma_full_temp_neg, beta)

        # Bd neg: True negative + (1 - True negative)*(No detection)
        Bdneg_ll = np.sum(
                         np.log(
                                (1 - neg.prev.values) +
                                neg.prev.values*nodetect_prob_neg
                                )
                         )

        # Bd pos: True positive * Detection * Load
        Bdpos_ll = np.sum((np.log(pos.prev.values) + np.log(detect_prob_pos) +
                           stats.norm(loc=pos.mean_load, scale=sigma_full_temp_pos).logpdf(np.log(pos.Bd_load.values))))

        # Bd likelihood
        ll = Bdneg_ll + Bdpos_ll

        # Abundance likelihood
        # dprob = 1 # params[np.where(np.array(param_priors.params) == 'detect_prob')[0][0]]
        # pred_abund = cdata.density.values*area*dprob # dprob and K are going to be hard to parse apart.
        # count_ll = np.sum(stats.poisson(pred_abund).logpmf(cdata.abund.values))
        # ll = ll + count_ll

        # Compute log priors of parameters with Jacobian adjustment
        lps = np.empty(len(params))
        for i, p in enumerate(param_priors.params):
            prm = params[i]
            l = param_priors.bounds[p][0]
            u = param_priors.bounds[p][1]

            # Jacobian accounts for sampling on the transformed space
            jac = np.log(np.abs(param_priors.jacobians[p](param_priors.transformations[p][0](prm,
                                                                            lower=l, upper=u),
                                                                            lower=l, upper=u)))
            lps[i] = param_priors.priors[p].logpdf(prm) + jac

        full_ll = np.sum(lps) + ll

    else:
        # If there are no individuals there should be no contribution to the
        # likelihood
        full_ll = 0
        ll = 0

    return((full_ll, ll))


def logistic_normal(mu, sigma, beta):
    """
    Approximation of the logistic-normal convolution from
    https://threeplusone.com/pubs/on_logistic_normal.pdf

    Parameters
    ----------
    mu : float
        Mean of the normal
    sigma : float
        Standard deviation of normal
    beta : float
        Slope of logistic

    Returns
    -------
    : float
        Detection probability given mu, sigma, and beta

    Note
    ----
    If logit(p) = alpha + beta * x, scale mu by mu + (alpha / beta)
    """
    gamma = np.sqrt(1 + (np.pi*sigma**2) / (8*(1 / beta)**2))
    return(1 / (1 + np.exp(-mu*(beta / gamma))))


def run_model(params, z_fxn, temp_fxn, temp_fxn_cv, temp_fxn_mm,
              initial_densities, model_params, steps, area,
              other_species, load_dependent_loss, start_date):
    """
    Run the IPM model

    Parameters
    ----------
    params : dict
        Keywords are species names
    z_fxn : fxn
        Survival function of zoospores in the environment
    temp_fxn : fxn
        How temperature is fluctuating at a site
    temp_fxn_cv : fxn
        How cv in temperature is fluctuating at a site
    temp_fxn_mm : fxn
        How the thermal mismatch is changing at a site
    initial_densities : dict
        Dictionary with keywords being species names and one keyword 'Z'
        that gives how many zoospores are in the pool. NOTE: These
        are actually estimated given the model parameters.
    model_params : dict
            Basic parameters for the model.
            'time_step' : The length of a time_step
    steps : int
        Number of time steps to simulate the model
    area : float
        The site area (e.g. area in m2)
    other_species : dict
        Dictionary with species parameters. Can be empty
    load_dependent_loss : bool
        Whether or not loss of infection depends on load
    start_date : pd.DateTime
        Start date of the simulation
    """

    focal_spps = list(params.keys())
    params.update(other_species)
    tsite = comm.Community(params, z_fxn,
                           initial_densities, model_params, area,
                           init_temp=z_fxn(temp_fxn(0)),
                           load_dependent=load_dependent_loss)
    trajs = tsite.simulate(steps, temp_fxn=temp_fxn,
                           temp_fxn_cv=temp_fxn_cv,
                           temp_fxn_mm=temp_fxn_mm)
    time_vals = np.arange(steps) * model_params['time_step']
    temperature_vals = temp_fxn(time_vals)

    # Format simulated data
    simdats = {}
    spps = list(params.keys())
    dates = start_date + pd.to_timedelta(time_vals, unit="Days")
    for i, spp in enumerate(spps):

        mean_loads = trajs[4*i + 3, :] / trajs[4*i + 2, :]
        dens = trajs[4*i + 2, :] + trajs[4*i + 1, :]
        prevs = trajs[4*i + 2, :] / dens
        simdats[spp] = pd.DataFrame({'date': dates,
                                     'mean_load': mean_loads,
                                     'density': dens,
                                     'prev': prevs,
                                     'temperature': temperature_vals})

    return({spp: simdats[spp] for spp in focal_spps})


def mcmc_metropolis(chain, seed, sppsite_dats, sppsite_counts, ppriors, base_params,
                    initial_densities, dod_location, site_numbers, start_date,
                    model_params, steps, temp_data, base_dat, longterm,
                    site_areas, site_areas_true, comm_site_params,
                    load_dependent_loss,
                    iters=5000, adapt=1000, tune=2.4, logger=None,
                    init_count=200):
    """
    Metropolis algorithm for estimating Bd infection parameters on species
    in sites.  Model assumes Bd growth parameters are the same across sites
    and uses all sites together to estimate parameters.  However, different
    dynamical models are fit to each site as species compositions could be
    different and temperature dynamics are different.

    Parameters
    ----------
    chain : int
        Chain number
    seed : float
        Random seed
    sppsite_dats : list of dicts
        Empirical data for each species for a given site. Keywords are species
        and items in the list represent different sites.
    sppsite_counts : list of dicts
        Same as sppsite dats, but contains observed abundances of species at
        particular sites for all survey dates. Zero abundances are included.
    ppriors : ABC model object
        Contains priors and all necessary transformations
    base_params : dict
        Contains the model parameters for the each amphibian species
    initial_densities : dict
        Contains the initial densities for each species
    dod_location : str
        The name of the dod location: 'LA', 'PA', 'VT', 'NM', or 'TN'
    site_numbers : array-like
        Numbers of sites to include.
    start_date : DateTime
        The date when the model starts
    model_params : dict
        Dict with the keyword 'time_step' that specifies the time step of the
        model.
    steps : int
        Number of steps to iterate the model
    temp_data : DataFrame
        HOBO temperature data
    base_dat : DataFrame
        Historical temperature data
    longterm : DataFrame
        20 year daily mean temperatures
    site_areas : array-like
        List of site areas
    site_areas_true: array-like
        Another list of site areas. If we are modeling on the m^2 scale then
        we use these site areas to scale.
    comm_site_params : list
        List of dictionaries. Each dictionary contains the non-focal species
        in a community (keywords are species names) and their fitted parameters.
        These will be included in the model (but not fit) to help capture the
        external zoospore pool. Only the focal species is fit.
    load_dependent_loss : bool
        Whether or not loss of infection should be considered load-dependent
    iters : int
        Number of MCMC iterations
    adapt : int
        At adapt time step, check acceptance rate and increase or decrease as
        needed.
    tune : float
        Initial proposal standard deviation from normal distribution

    Returns
    -------
    : dict
        'params': Parameters at each step step
        'posterior': Log posterior up to a normalizing constant
        'acceptance': Acceptance vector
        'loglik': Log-likelihood of model given data.

    Notes
    -----

    The Adaptive MCMC proceeds as follows. Sample with independent normals with
    parameter `tune` for the first 100 steps. Then update tuning parameter to
    try and get closer to the optimal acceptance rate.  Periodically throughout
    this warm-up update, compute
    the covariance for the parameters and use a multivariate
    normal proposal with mu = current parameters and sigma = cov. Continue to tune
    to to get close to the optimal acceptance rate. Finally, start sampling.

    """

    # All species are included in every site.
    spps = list(sppsite_dats[0].keys())

    all_params = {spp: np.empty((len(ppriors[spp].params), iters + 1)) for spp in spps}
    all_trans_params = {spp: np.empty((len(ppriors[spp].params), iters + 1)) for spp in spps}
    all_logpost = np.empty(iters + 1)
    all_loglik = np.empty(iters + 1)

    # Choose random starting parameters

    sum_init_post = -np.inf
    counter = 0

    f = "Initializing sampler..."
    if logger is None:
        print(f)
    else:
        logger.info(f)
    while (sum_init_post == -np.inf):

        if counter >= init_count:
            raise AssertionError("Can't find adequate starting value")

        init_params = {spp: {pnm: ppriors[spp].priors[pnm].rvs(size=seed)[seed - 1]
                             for pnm in ppriors[spp].params}
                       for spp in spps}

        init_params_vect = {spp: np.array([init_params[spp][p] for p in ppriors[spp].params]) for spp in spps}

        for spp in spps:
            all_params[spp][:, 0] = init_params_vect[spp]

        # Initialize
        full_init_params = {spp: base_params[spp].copy() for spp in spps}
        [full_init_params[spp].update(init_params[spp]) for spp in spps]

        # Different temperatures for different sites
        temp_fxns = [build_temperature_fxn(temp_data, base_dat,
                                           longterm, sitenum,
                                           start_date) for sitenum in site_numbers]

        z_fxn = comm.zsurv_fxn(model_params['time_step'])

        simdats_sites = [run_model(full_init_params, z_fxn, temp_fxn, temp_fxn_cv,
                                   temp_fxn_mm, initial_densities, model_params,
                                   steps, area, other_species, load_dependent_loss, start_date)
                         for (temp_fxn, temp_fxn_cv, temp_fxn_mm), area, other_species in
                         zip(temp_fxns, site_areas, comm_site_params)]

        # Compute posterior for species and sites
        init_post = []
        init_ll = []
        for simdats, sppsite_dat, sppsite_count, area in zip(simdats_sites, sppsite_dats, sppsite_counts, site_areas_true):
            for spp in spps:
                full_ll, ll = logpost(sppsite_dat[spp], sppsite_count[spp], simdats[spp],
                                      init_params_vect[spp], ppriors[spp], area)
                init_post.append(full_ll)
                init_ll.append(ll)

        sum_init_post = np.sum(init_post)
        sum_init_ll = np.sum(init_ll)
        counter += 1

    f = "Initialization complete"
    if logger is None:
        print(f)
    else:
        logger.info(f)

    all_logpost[0] = sum_init_post
    all_loglik[0] = sum_init_ll # Save log-likelihood for DIC comparison

    # Metropolis algorithm
    acceptance = np.zeros(iters)

    # Initial tuning parameter
    tune_sigmas = {}
    for spp in spps:
        c = all_params[spp].shape[0]
        tune_sigmas[spp] = (tune**2 / c) * np.eye(c)

    for step in range(1, iters + 1):

        # Log progress
        if step % 10 == 0:

            f = "Chain {0}, {1}".format(chain, step)
            if logger is None:
                print(f)
            else:
                logger.info(f)

        current_params = {spp: all_params[spp][:, step - 1] for spp in spps}
        current_logpost = all_logpost[step - 1]
        current_ll = all_loglik[step - 1]
        current_trans_params = {spp: np.array([ppriors[spp].transformations[pnm][0](np.array([i]),
                                                                                    lower=ppriors[spp].bounds[pnm][0],
                                                                                    upper=ppriors[spp].bounds[pnm][1])
                                               for pnm, i in
                                               zip(ppriors[spp].params, current_params[spp])]).ravel()
                                for spp in spps}

        # Save transformed parameters
        for spp in spps:
            all_trans_params[spp][:, step - 1] = current_trans_params[spp]

        # Update the tuning every 100 steps
        check1 = np.int(adapt / 2)
        check2 = 100 # Tune every 100 steps

        if (step < (5*adapt)) & ((step % check2) == 0) & (step > check1):

            arate = (np.sum(acceptance[(step - check2):step]) / check2)

            # Update tuning parameter to get the acceptance rate in the
            # sweet spot
            update = False
            old_tune = tune
            if arate > 0.5:
                tune = old_tune * 1.2
                update = True
            elif arate < 0.15:
                tune = old_tune / 1.2
                update = True

            if update:
                if logger is None:
                    print("New tuning parameter is {2:.2} -> {0:.2} and arate is {1:.2}".format(tune, arate, old_tune))
                else:
                    logger.info("New tuning parameter is {2:.2} -> {0:.2} and arate is {1:.2}".format(tune, arate, old_tune))
            else:
                if logger is None:
                    print("No update. Tuning parameter is {0:.2} and arate is {1:.2}".format(tune, arate))
                else:
                    logger.info("No update. Tuning parameter is {0:.2} and arate is {1:.2}".format(tune, arate))

            # Update the tuning parameter
            tune_sigmas = {spp: tune**2 * (tune_sigmas[spp] / (old_tune**2)) for spp in spps}

            # Update MCMC with parameter variance to help mixing
            if (step == 2*adapt) or (step == 3*adapt) or (step == 4*adapt):

                if (step == 2*adapt) or (step == 3*adapt) or (step == 4*adapt):
                    tune = 2.4  # Based an Gelman et al. 2.4 is the theoretical ideal.

                for spp in spps:
                    c = all_params[spp].shape[0]
                    # Need to compute variance on the transformed parameters
                    tune_sigmas[spp] = (tune**2 / c) * np.cov(all_trans_params[spp][:, (step - adapt):(step)])

                    if logger is None:
                        print("Updated parameter variance")
                    else:
                        logger.info("Updated parameter variance")

        # Random normal proposal
        proposed_trans_params = {spp: np.random.multivariate_normal(
                                                 mean=current_trans_params[spp],
                                                 cov=tune_sigmas[spp])
                                 for spp in spps}

        # Untransform parameters for model
        proposed_params = {spp: np.array([ppriors[spp].transformations[pnm][1](np.array([i]), lower=ppriors[spp].bounds[pnm][0], upper=ppriors[spp].bounds[pnm][1]) for pnm, i in
                                         zip(ppriors[spp].params, proposed_trans_params[spp])]).ravel()
                           for spp in spps}

        # Get log posterior
        full_params = {spp: base_params[spp].copy() for spp in spps}
        [full_params[spp].update({pnm: p for pnm, p in zip(ppriors[spp].params, proposed_params[spp])}) for spp in spps]

        # Loop over different temperatures at different sites
        simdats_sites = [run_model(full_params, z_fxn, temp_fxn, temp_fxn_cv,
                                   temp_fxn_mm, initial_densities,
                                   model_params, steps, area,
                                   other_species, load_dependent_loss, start_date)
                         for (temp_fxn, temp_fxn_cv, temp_fxn_mm), area, other_species in
                         zip(temp_fxns, site_areas, comm_site_params)]

        # Loop over species and sites
        prop_post = []
        prop_ll = []
        for simdats, sppsite_dat, sppsite_count, area in zip(simdats_sites, sppsite_dats, sppsite_counts, site_areas_true):
            for spp in spps:
                full_ll, ll = logpost(sppsite_dat[spp], sppsite_count[spp], simdats[spp],
                                      proposed_params[spp], ppriors[spp], area)
                prop_post.append(full_ll)
                prop_ll.append(ll)

        proposed_logpost = np.sum(prop_post)
        proposed_ll = np.sum(prop_ll)

        # Reject or accept
        logr = proposed_logpost - current_logpost # nan evaluates to true

        logu = np.log(np.random.random())
        if logu < logr:

            for spp in spps:
                all_params[spp][:, step] = proposed_params[spp]

            all_logpost[step] = proposed_logpost
            all_loglik[step] = proposed_ll
            acceptance[step - 1] = 1
        else:

            for spp in spps:
                all_params[spp][:, step] = current_params[spp]

            all_logpost[step] = current_logpost
            all_loglik[step] = current_ll

            acceptance[step - 1] = 0

        if step % 100 == 0:
            accept = acceptance[(step - 100):step].sum() / 100
            f = "Chain {0}: Acceptance over the last 100: {1}".format(chain, accept)

            if logger is None:
                print(f)
            else:
                logger.info(f)

    return({'params': all_params,
            'posterior': all_logpost,
            'loglik': all_loglik,
            'acceptance': acceptance})


def multiprocess_mcmc(processes, chains, sppsite_dats, sppsite_counts, ppriors, base_params,
                      initial_densities, dod_location, sitenum, start_date,
                      model_params, steps, temp_data, base_dat, longterm,
                      site_areas, site_areas_true, comm_site_params,
                      load_dependent_loss, iters, adapt,
                      tune, logger):
    """
    Multiprocess mcmc to run multiple chains
    """

    if processes <= 1:

        chain_num = 0
        seed = 3 #np.random.randint(1, 10000)
        results = [mcmc_metropolis(chain_num, seed, sppsite_dats, sppsite_counts, ppriors, base_params,
                                   initial_densities, dod_location, sitenum, start_date,
                                   model_params, steps, temp_data, base_dat,
                                   longterm, site_areas, site_areas_true, comm_site_params,
                                   load_dependent_loss,
                                   iters, adapt, tune, logger)]

    else:

        pool = mp.Pool(processes=processes)
        results = [pool.apply_async(mcmc_metropolis, args=(i, seed, sppsite_dats, sppsite_counts, ppriors, base_params,
                                                           initial_densities, dod_location, sitenum, start_date,
                                                           model_params, steps, temp_data, base_dat, longterm,
                                                           site_areas, site_areas_true, comm_site_params,
                                                           load_dependent_loss,
                                                           iters, adapt, tune, logger))
                   for i, seed in enumerate(np.random.randint(1, 10000, size=chains))]

        results = [p.get() for p in results]
        pool.close()

    return(results)


def get_temp_data(dod_location, average_sites=False, fix_min=True):
    """
    Load the HOBO temperature data, historical data for a given DOD location,
    and the long-term average temperature across sites

    Parameters
    ----------
    dod_location : str
        DOD location
    average_sites : bool
        If True, average temperature across sites
    fix_mine : bool
        If True, temperature can't dip below 4 C.

    Returns
    -------
    : tuple
        (HOBO data , base temperature data, long-term mean)
    """

    # HOBO temperature data
    temp_data = pd.read_csv(os.path.join(temperature_path,
                                         dod_location,
                                         "hobo_water_temperature.csv"))

    # Historical temperature data
    base_dat = pd.read_csv(os.path.join(base_temp_path,
                           "{0}_1999-2019_temperatures.csv".format(dod_location)))
    base_dat = base_dat.assign(datetime=lambda x: pd.to_datetime([p.split("T")[0] for p in x.unix_time]))
    base_dat = base_dat.assign(month=lambda x: x.datetime.dt.month,
                               day=lambda x: x.datetime.dt.day)

    # Because this is air_temperature that we are approximating for water temperature
    # don't let it go below 4 C, which is hydrologically the coldest standing
    # water really gets.
    if fix_min:
        base_dat.loc[base_dat.mean_temperature < 4, "mean_temperature"] = 4

    # Long-term average temperature data
    longterm = pd.read_csv(os.path.join(base_temp_path, "{0}_longterm_avg_temperature.csv".format(dod_location)))
    longterm.loc[:, 'datetime'] = pd.to_datetime([p.split("T")[0] for p in longterm.datetime])

    temp_data = temp_data.assign(datetime=lambda x: pd.to_datetime(x.datetime))

    if average_sites:
        temp_data = temp_data.assign(day=lambda x: x.datetime.dt.day,
                                     month=lambda x: x.datetime.dt.month,
                                     year=lambda x: x.datetime.dt.year)
        temp_data = (temp_data.groupby(['dod_location', 'day', 'month', 'year'])
                              .agg({'water_temperature_C': np.mean})
                              .reset_index()
                              .assign(datetime=lambda x: [pd.datetime(yr, mn, d) for yr, mn, d in zip(x.year, x.month, x.day)],
                                      site=1)
                              .sort_values(by=['datetime']))
        base_dat = (base_dat.groupby(['DOD_location', 'datetime'])
                            .agg({'mean_temperature': np.mean})
                            .reset_index()
                            .sort_values(by=['datetime'])
                            .assign(site_number=1))
        longterm = (longterm.groupby(['DOD_location', 'month', 'day', 'datetime'])
                            .agg({'mean_temperature': np.mean})
                            .assign(site_number=1)
                            .reset_index()
                            .sort_values(by=['datetime']))

    return((temp_data, base_dat, longterm))


def format_obs(obs_dat):
    """
    Format observed data for plotting comparison
    """

    def aggfxn(x):

        names = {
            'prev': x.Bd_pos.mean(),
            'mean_load': np.log(x[x.Bd_load > 0].Bd_load).mean()
        }

        return(pd.Series(names, index=['prev', 'mean_load']))

    gobs = obs_dat.groupby(['date']).apply(aggfxn).reset_index()
    return(gobs)


def get_fitted_parameters(spp, dod_location, model):
    """
    Get and return the dictionary of species parameters for a specific location
    base on model fit to all sites with temperature-dependent Bd growth on
    a host. Sets fit_mu0 to False

    Parameters
    ----------
    spp : str
        Species name
    dod_location : str
        DOD location name
    model : str
        Model name. 'a_temp', 'trans_beta_temp', 'null_model', 'lI_temp'

    Returns
    -------
    : dict
        Dictionary of model parameters for species dod location
    """

    tres = pd.read_pickle("../results/pickled_results/{0}_{1}_{2}_allsites_w_K.pkl".format(dod_location, model, spp))
    base_params = yaml.safe_load(open("model_params/spp_params_{0}.yml".format(spp), 'r'))['params']
    burnin = tres['adapt_param']*4
    tchains = tres['chains']
    fitparams = np.array([tres['mcmc_results'][i]['params'][spp][:, burnin:].T
                          for i in range(tchains)])
    tbp = {p: val for p, val in zip(tres['parameters'],
                                    fitparams.mean(axis=(0, 1)))}
    base_params.update(tbp)
    base_params['fit_mu0'] = False
    return(base_params)


def get_count_data(dod_dat):
    """
    Get the count data for all visits and all species
    """

    def date_group(x):

        names = {
            'abund': len(x.Date)
        }

        return(pd.Series(names, index=['abund']))

    abund_dat = (dod_dat.groupby(['Date', 'Site_code', 'Species_code'])
                        .apply(date_group)
                        .reset_index())

    full_abund = abund_dat.pivot_table(index=["Date", 'Site_code'],
                                       columns='Species_code',
                                       values="abund",
                                       fill_value=0).reset_index()

    # full_abund = full_abund.rename(columns={'Date': 'date'})

    return(full_abund)


if __name__ == '__main__':

    # Load Bd data
    datapath = os.path.join(data_dir, "leopard_frog_data.csv")

    # Exclude parts of the data to test sensitivity
    # [('LA', pd.datetime(2017, 6, 1), [1, 2, 3, 4, 5])])
    exclude = None

    fulldat = load_and_format_data(datapath, exclude=exclude)
    site_info = pd.read_csv("../data/site_info.csv")

    # DOD location, external temperature, omega value
    dod_options = ['LA', 'TN', 'PA', 'VT']

    # Is there temperature variation in the external contribution?, amount of external contribution omega
    etemp_options = [(False, np.log(1))]

    # Some alternatives.

    # (False, np.log(500)),
    # (True, np.log(500)),
    # (False, -100000)]

    # Loop through model criteria
    mcmc_loops = [(tdod, etemp, tomega) for tdod in dod_options
                  for etemp, tomega in etemp_options]

    for dod_location, external_temp, omega in mcmc_loops:

        site_level_params = yaml.safe_load(open("model_params/{0}_params/{0}_site_level_params.yml".format(dod_location, "r")))

        # Boolean options
        # external_source : Do you fit omega?
        # fit_with_beta_prior : Do you use an informative prior for trans_beta when fitting?
        external_source = True  # Do you include an external source of zoospores?
        fit_with_beta_prior = True  # Use an informative prior on trans_beta?

        dod_dat = fulldat[(fulldat.DOD_location == dod_location) &
                          (fulldat.Site_code.isin(site_level_params['sites']))]
        obs_dat = dod_dat[['Site_code', 'Species_code', 'Date', 'Bd_pos', 'Bd_load']]
        site_dat = site_info[site_info.DOD_location == dod_location]

        count_dat = get_count_data(dod_dat)

        # Loop it here
        species_vect = [([spp], sites) for spp, sites in
                        site_level_params['spp'].items()]

        # Determines host density
        Kvals = np.array([4, 8, 10])

        # Loop through different density assumptions
        for tK in Kvals:
            for tspp_vect, tsite_numbers in species_vect:

                # Build species-level data
                spps = tspp_vect
                full_site_numbers = tsite_numbers
                site_numbers = full_site_numbers

                # Bd prevalence and load data
                sppsite_dats = [{spp: (obs_dat.query("Site_code == {0} and Species_code == '{1}'".format(sitenum, spp))
                                              .assign(date=lambda x: pd.to_datetime(x.Date))
                                              .sort_values(by=['date'])
                                              .reset_index(drop=True)) for spp in spps}
                                for sitenum in site_numbers]

                # Species count data
                sppsite_counts = [{spp: (count_dat.query("Site_code == {0}".format(sitenum))
                                                  .assign(date=lambda x: pd.to_datetime(x.Date))
                                                  .sort_values(by=['date'])
                                                  .reset_index(drop=True)[['date', spp]]
                                                  .rename(columns={spp: 'abund'})) for spp in spps}
                                  for sitenum in site_numbers]

                # These are only used if abundance data are fit. It is ok if
                # they are nan if abundance data is not fit.
                site_areas_true = [site_dat[site_dat.site_number == site].area_m2.values[0]
                                   for site in site_numbers]

                # Model on the m2 scale
                site_areas = [1.0 for site in site_numbers]

                # Use as a place-holder because we are only considering one
                # species
                comm_site_params_full = [{} for i in range(len(site_areas))]

                # Get the temperature data
                temp_data, base_dat, longterm = get_temp_data(dod_location,
                                                              average_sites=False)

                ##### Set up simulation #######

                steps = 201 # Length of model simulation
                start_date = pd.datetime(2016, 1, 1) # Start date of model simulation
                base_params = {spp: yaml.safe_load(open("model_params/{0}_params/spp_params_{1}.yml".format(dod_location, spp), 'r'))['params'] for spp in spps}
                model_params = {'time_step': 7, 'mean_temp': 15}

                # Model to fit for each species:
                # (temperature param,
                #  temp mismatch param,
                #  Bool on whether mu0 should be fit,
                #  Bool for whether loss of infection depends on load)

                # null_model : Equivalent to "Demography-dependent transmission"
                # trans_beta_temp : Equivalent to "Temperature-dependent Bd transmission"
                # a_temp : Equivalent to "Temperature-dependent Bd growth"
                # both_model : Equivalent to "Temperature-dependent Bd growth and transmission"
                fit_params = [('null_model', '', False, False),
                              ('trans_beta_temp', 'trans_beta_mm', False, False),
                              ('a_temp', 'a_mm', False, True),
                              ('both_model', '', False, True)]

                # Loop over different temperature scenarios
                for temp_param, mm_param, fit_mu0, load_dependent_loss in fit_params:

                    mod_name = temp_param
                    logger.info("Starting {0}, {1}".format(spps, mod_name))

                    load_param = 'a'
                    for spp in base_params.keys():

                        # Unidentifiable parameter between 0-1.
                        # Use lab data to inform
                        base_params[spp]['b'] = 0.5

                        # Check if there is a default b, and reset it so
                        if 'fixed_params' in site_level_params:
                            if 'b' in site_level_params['fixed_params']:
                                base_params[spp]['b'] = site_level_params['fixed_params']['b']

                        base_params[spp]['sI'] = 1  # Fit with sI = 1
                        base_params[spp]['fit_mu0'] = False  # Do not try to fit_mu0

                    # Empty community parameters. Only relevant if we use other species
                    comm_site_params = [{} for i in range(len(comm_site_params_full))]
                    initial_densities = {spp: np.array([0, 1.0, 1.0, 1.0]) for spp in spps}
                    initial_densities['Z'] = np.array([10.0])

                    if load_dependent_loss:
                        base_est_params = ['mu_l', load_param, 'trans_beta',
                                           'sigma_full']
                    else:
                        base_est_params = ['lI', load_param, 'trans_beta',
                                           'sigma_full']

                    # Set carrying capacity
                    for spp in base_params.keys():
                        base_params[spp]['K'] = tK

                    # Do we try to fit omega?
                    if external_source:
                        base_est_params = base_est_params + ['omega']
                        for spp in base_params.keys():
                            base_params[spp]['omega_temp'] = 0 # Zoospores contributed at mean_temp
                    else:
                        for spp in base_params.keys():
                            base_params[spp]['omega'] = omega # Zoospores contributed at mean_temp

                            if external_temp:
                                base_params[spp]['omega_temp'] = -0.5
                            else:
                                base_params[spp]['omega_temp'] = 0  #-0.5 # Zoospores contributed at mean_temp

                    # Only used if "both_model" is the model type
                    temp_param2 = 'trans_beta_temp'

                    if temp_param == 'null_model':
                        est_params = base_est_params
                    elif temp_param == 'both_model':
                        temp_param = 'a_temp'
                        est_params = base_est_params + [temp_param, temp_param2]
                    else:
                        est_params = base_est_params + [temp_param]

                    # Set priors on untransformed parameters
                    priors = {'mu_l': stats.norm(loc=0, scale=2),
                              'sigma_l': stats.halfnorm(scale=1), # Not fitting this parameter anymore
                              'lI': stats.norm(loc=0, scale=3),
                              load_param: stats.norm(loc=5, scale=5),
                              'sigma_full': stats.halfnorm(scale=2),
                              'trans_beta': stats.halfnorm(scale=1),  # stats.uniform(loc=0, scale=0.06), #
                              temp_param: stats.norm(loc=0, scale=1),
                              temp_param2: stats.norm(loc=0, scale=1),
                              'omega': stats.uniform(loc=0, scale=10)}

                    # Specify the transformations to put parameters on the
                    # unconstrained scale
                    transformations = {'mu_l': None,
                                       'sigma_l': 'log',
                                       'lI': None,
                                       load_param: None,
                                       'trans_beta': 'log',
                                       'sigma_full': 'log',
                                       'K': None,
                                       'omega': 'minmax',
                                       temp_param: None,
                                       temp_param2: None}

                    # Update priors if site has species-specific priors
                    # From previous experiments, for example or when constraining trans_beta
                    if "priors" in site_level_params:
                        for p in site_level_params['priors'].keys():
                            priors[p] = eval(site_level_params['priors'][p][0])

                            # Update
                            if site_level_params['priors'][p][1] == 'None':
                                transformations[p] = None
                            else:
                                transformations[p] = site_level_params['priors'][p][1]

                    # A model fit with an information prior on trans_beta
                    if fit_with_beta_prior:

                        # Different models can have different trans_beta priors
                        if load_dependent_loss:
                            loss_ind = "loss_load"
                        else:
                            loss_ind = "no_loss_load"

                        for p in site_level_params['priors_K={0}'.format(tK)][loss_ind].keys():
                            priors[p] = eval(site_level_params['priors_K={0}'.format(tK)][loss_ind][p][0])
                            transformations[p] = site_level_params['priors_K={0}'.format(tK)][loss_ind][p][1]

                    ppriors = {spp: abc.ABC(None, est_params, None, None,
                                            transformations=transformations,
                                            priors=priors) for spp in spps}

                    ##### Initialize MCMC ####

                    # Note that the number of adaptive steps and and iterations
                    # may need to be updated based on model convergence
                    iters = 20000  # This includes 5x adapt warmup samples.
                    adapt = 2000  # 5x adapt is the warmup
                    tune = 0.7  # Initial tuning. Will be updated adaptively.
                    chains = 4  # MCMC chains
                    processes = 4  # Number of cores to use

                    # Fit model
                    mcmc_res = multiprocess_mcmc(processes, chains, sppsite_dats, sppsite_counts, ppriors,
                                                 base_params, initial_densities, dod_location,
                                                 site_numbers, start_date, model_params, steps,
                                                 temp_data, base_dat, longterm, site_areas, site_areas_true,
                                                 comm_site_params, load_dependent_loss,
                                                 iters=iters, adapt=adapt,
                                                 tune=tune, logger=logger)

                    ##### Check model diagnostics  #####

                    fig, tax = plt.subplots(1, 1)
                    for chain in range(chains):

                        all_params = mcmc_res[chain]['params']
                        spp = spps[0]
                        for i in np.arange(len(ppriors[spp].params))[2:3]:
                            pnm = ppriors[spp].params[i]
                            # tax.plot(ppriors[spp].transformations[pnm][0](all_params[spp][i, :],
                            #                                               *ppriors[spp].bounds[pnm]), label=ppriors[spp].params[i])
                            tax.plot(all_params[spp][i, :], label=ppriors[spp].params[i])

                    plt.legend(loc=(1, 0.5), prop={'size': 5})
                    plt.show()

                    ndarray = np.array([mcmc_res[i]['params'][spp][:, 5*adapt:].T for i in range(chains)])
                    dataset = az.convert_to_inference_data(ndarray)
                    rhat = az.rhat(dataset, method="rank")
                    rhat_diag = np.all(np.array(rhat.to_array()) < 1.01)
                    ess = az.ess(dataset, method="bulk")
                    ess_diag = np.all(np.array(ess.to_array()) > 400)
                    az.plot_ess(dataset, kind="evolution", figsize=(12, 6))
                    az.plot_rank(dataset, figsize=(10, 6))

                    ##### Check model fit to data (for each site and species) #####

                    # 1. Draw sampled params
                    est_spp_params = {}
                    sims = 10
                    for spp in spps:

                        ndarray = np.array([mcmc_res[i]['params'][spp][:, 5*adapt:].T for i in range(chains)])
                        num_samp = ndarray.shape[1]
                        samp_params = []

                        # Draw multiple copies of parameters
                        for num in range(sims + 1):

                            if num < sims:
                                samp_params.append({ep: ndarray[np.random.randint(0, chains),
                                                                np.random.randint(0, num_samp), i]
                                                    for i, ep in enumerate(est_params)})
                            else:
                                # Get mean parameters
                                mean_params = ndarray.mean(axis=(0, 1))
                                samp_params.append({ep: mean_params[i]
                                                    for i, ep in enumerate(est_params)})


                        est_spp_params[spp] = samp_params

                    # 2. Simulate the model
                    z_fxn = comm.zsurv_fxn(model_params['time_step'])
                    sim_res_site = []
                    for s, site in enumerate(site_numbers):

                        # Get temperature functions
                        temp_fxns = build_temperature_fxn(temp_data, base_dat, longterm, site,
                                                          start_date)
                        temp_fxn, temp_fxn_cv, temp_fxn_mm = temp_fxns

                        all_sim_res = []
                        for sim_num in range(sims + 1):

                            params = {}
                            for spp in spps:

                                tparams = base_params[spp].copy()
                                tparams.update(est_spp_params[spp][sim_num])
                                params[spp] = tparams

                            sim_res = run_model(params, z_fxn, temp_fxn, temp_fxn_cv,
                                                temp_fxn_mm, initial_densities, model_params,
                                                steps, site_areas[s], comm_site_params[s],
                                                load_dependent_loss, start_date)
                            all_sim_res.append(sim_res)

                        sim_res_site.append(all_sim_res)

                    # 2.5 Compute log-likelihood for best Bayesian parameters.
                    best_post = []
                    best_ll = []
                    for simdats, sppsite_dat, sppsite_count, area in zip(sim_res_site, sppsite_dats, sppsite_counts, site_areas_true):
                        for spp in spps:
                            best_params = np.array([est_spp_params[spp][sims][p]
                                                    for p in ppriors[spp].params])
                            full_ll, ll = logpost(sppsite_dat[spp], sppsite_count[spp], simdats[sims][spp],
                                                  best_params, ppriors[spp], area)
                            best_post.append(full_ll)
                            best_ll.append(ll)

                    # Compute model DIC
                    best_loglik = np.sum(best_ll)
                    all_loglik = np.concatenate([mcmc_res[i]['loglik'][5*adapt: ] for i in range(chains)])
                    pDIC = 2*(best_loglik - np.mean(all_loglik)) # Gelman BDA3 pg. 172
                    pDIC_alt = 2*np.var(all_loglik)
                    DIC = -2*best_loglik + 2*pDIC
                    DIC_alt = -2*best_loglik + 2*pDIC_alt

                    # Save results
                    suffix = ""

                    if external_source:
                        suffix = suffix + "_ext"

                    if external_temp:
                        suffix = suffix + "_var_temp"

                    if fit_with_beta_prior:
                        suffix = suffix + "_beta_prior"

                    if exclude is not None:
                        suffix = suffix + "_exclude"

                    if load_dependent_loss:
                        suffix2 = "_loss_load"
                    else:
                        suffix2 = ""

                    pd.to_pickle({'species': spps, 'sites': site_numbers,
                                  'adapt_param': adapt,
                                  'parameters': est_params,
                                  'chains': chains,
                                  'mcmc_results': mcmc_res,
                                  'DIC_values': (DIC, DIC_alt),
                                  'diagnostics': (rhat_diag, ess_diag),
                                  'base_params': base_params},
                                 "../results/pickled_results/{0}_{1}_{2}_allsites{3}_K={4}{5}{6}.pkl".format(dod_location, mod_name, "-".join(spps), suffix, tK, suffix2, "_omega=" + str(np.int(np.round(np.exp(omega), decimals=0)))))

                    # 3. Compare to observed data
                    labsize = 6
                    ms = 4
                    for spp in spps:

                        fig, axes = plt.subplots(len(site_numbers), 2,
                                                 figsize=(8, 1.3*len(site_numbers)),
                                                 sharex=True, sharey=False)
                        axes = np.atleast_2d(axes)

                        for s, site in enumerate(site_numbers):

                            # Plot simulation data
                            for sim_num in range(sims):
                                tsim_dat = sim_res_site[s][sim_num][spp]

                                axes[s, 0].plot(tsim_dat.date, tsim_dat.prev, '-', color='gray', alpha=0.4)
                                axes[s, 1].plot(tsim_dat.date, tsim_dat.mean_load, '-', color="gray", alpha=0.4)
                                # axes[s, 2].plot(tsim_dat.date, tsim_dat.density*site_areas_true[s], '-', color="gray", alpha=0.4)

                            # Plot observed data
                            tobs_dat = format_obs(sppsite_dats[s][spp])
                            tcount_dat = sppsite_counts[s][spp]

                            try:
                                axes[s, 0].plot(tobs_dat.date, tobs_dat.prev, '-o', ms=ms)
                                axes[s, 1].plot(tobs_dat.date, tobs_dat.mean_load, '-o', ms=ms)
                                # axes[s, 2].plot(tcount_dat.date, tcount_dat.abund, '-o', ms=ms)
                            except AttributeError:
                                pass

                            axes[s, 1].set_ylabel("ln Bd load, site {0}".format(site), size=labsize)
                            axes[s, 0].set_ylabel("Bd prev, site {0}".format(site), size=labsize)
                            # axes[s, 2].set_ylabel("Host density, site {0}".format(site), size=labsize)

                            if s == 0:
                                axes[s, 0].set_title(spp)
                                axes[s, 1].set_title(spp)

                                # Set up legend
                                markers = ['o', '']
                                colors = [sns.color_palette()[0], 'gray']
                                labels = ['obs.', 'pred.']
                                handles = [plt.Line2D([0], [0], marker=tm, linestyle='-', color=tc, label=tl)
                                           for tm, tc, tl in zip(markers, colors, labels)]
                                axes[s, 0].legend(handles=handles,
                                                  prop={'size': labsize}, frameon=False,
                                                  framealpha=0.5)

                            if axes[s, 1].is_last_row():
                                axes[s, 1].set_xlabel("Date", size=labsize)
                                axes[s, 0].set_xlabel("Date", size=labsize)
                                # axes[s, 2].set_xlabel("Date", size=labsize)

                            for pi in range(2):
                                axes[s, pi].tick_params(labelsize=labsize)

                        plt.tight_layout()

                        fig.savefig("../results/{0}_{1}_{2}_allsites_obs_pred{3}_{4}_K={5}{6}{7}.png".format(dod_location, spp,
                                                                                             mod_name, suffix,
                                                                                             "-".join(spps), tK, suffix2, "_omega=" + str(np.int(np.round(np.exp(omega), decimals=0)))))
                        plt.close("all")

