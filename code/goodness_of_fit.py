import os
import pandas as pd
import reduced_model as comm
import numpy as np
import scipy.stats as stats
import reduced_mcmc as red_mcmc

"""
Run goodness of fit tests for best fit models on observed Bd prevalence and
intensity.

"""
# Set up file paths
home_path = os.path.expanduser("~")
data_dir = os.path.join("..", "data")
base_temp_path = os.path.join(data_dir,
                              "temperature_data",
                              "historical_temperature_data")
temperature_path = os.path.join(data_dir,
                                "temperature_data",
                                "measured_temperature_data")

if __name__ == '__main__':

    # Set preliminary parameters

    # Site by species dict
    spp_dict = {'PA': 'LIPI', 'VT': 'LIPI', 'LA': "LISP", 'TN': 'LISP'}

    # Best fit models for each location
    models = {'PA': 'a_temp', 'VT': 'a_temp',
              'TN': 'a_temp', 'LA': 'both_model'}
    load_dependent_loss = True  # All best models have load-dependent loss

    rsq_ests = {}
    percent_pvals = {}
    for dod_location in ['VT', 'PA', "TN", 'LA']:

        # Extract model fits
        res = pd.read_pickle("../results/pickled_results/{0}_{2}_{1}_allsites_ext_beta_prior_K=8_loss_load_omega=1.pkl".format(dod_location, spp_dict[dod_location], models[dod_location]))
        spps = res['species']
        site_numbers = res['sites']
        adapt = res['adapt_param']
        est_params = res['parameters']
        chains = res['chains']
        mcmc_res = res['mcmc_results']
        base_params = res['base_params']

        # Set simulation parameters
        model_params = {'time_step': 7, 'mean_temp': 15}
        initial_densities = {spp: np.array([0, 1.0, 1.0, 1.0]) for spp in spps}
        comm_site_params = [{} for i in range(len(site_numbers))]
        site_areas = [1.0 for site in site_numbers]
        start_date = pd.datetime(2016, 1, 1)
        steps = 201

        # Set up data for simulation and goodness of fit
        datapath = os.path.join(data_dir, "leopard_frog_data.csv")
        fulldat = red_mcmc.load_and_format_data(datapath, exclude=None)
        dod_dat = fulldat[(fulldat.DOD_location == dod_location) &
                          (fulldat.Site_code.isin(site_numbers))]
        obs_dat = dod_dat[['Site_code', 'Species_code', 'Date', 'Bd_pos', 'Bd_load']]
        sppsite_dats = [{spp: (obs_dat.query("Site_code == {0} and Species_code == '{1}'".format(sitenum, spp))
                                      .assign(date=lambda x: pd.to_datetime(x.Date))
                                      .sort_values(by=['date'])
                                      .reset_index(drop=True)) for spp in spps}
                        for sitenum in site_numbers]

        # Extract temperature data
        temp_data, base_dat, longterm = red_mcmc.get_temp_data(dod_location,
                                                               average_sites=False)

        # 1. Get median parameter estimates
        est_spp_params = {}
        sims = 1
        for spp in spps:

            ndarray = np.array([mcmc_res[i]['params'][spp][:, 5*adapt:].T for i in range(chains)])
            num_samp = ndarray.shape[1]
            samp_params = []

            # Extract median parameters for prediction
            for num in range(sims):

                mean_params = np.median(ndarray, axis=(0, 1))
                samp_params.append({ep: mean_params[i]
                                    for i, ep in enumerate(est_params)})

            est_spp_params[spp] = samp_params

        # 2. Simulate the model
        z_fxn = comm.zsurv_fxn(model_params['time_step'])
        sim_res_site = []
        for s, site in enumerate(site_numbers):

            # Get temperature functions
            temp_fxns = red_mcmc.build_temperature_fxn(temp_data, base_dat, longterm, site,
                                                       start_date)
            temp_fxn, temp_fxn_cv, temp_fxn_mm = temp_fxns

            all_sim_res = []
            for sim_num in range(sims):

                params = {}
                for spp in spps:

                    tparams = base_params[spp].copy()
                    tparams.update(est_spp_params[spp][sim_num])
                    params[spp] = tparams

                sim_res = red_mcmc.run_model(params, z_fxn, temp_fxn, temp_fxn_cv,
                                             temp_fxn_mm, initial_densities, model_params,
                                             steps, site_areas[s], comm_site_params[s],
                                             load_dependent_loss, start_date)
                all_sim_res.append(sim_res)

            sim_res_site.append(all_sim_res)

        # 3. Compute goodness of fit
        gof_dt = []
        for s in range(len(site_numbers)):
            for j in range(sims):

                odat = sppsite_dats[s][spps[0]]
                simdat = sim_res_site[s][j][spps[0]]
                data = pd.merge_asof(odat, simdat, direction="nearest", on='date')
                gof_dt.append(data)

        gof_dt = pd.concat(gof_dt)

        # Compute R2 using formula from Gelman et al. 2017
        ind = gof_dt.Bd_load > 0
        rsq_dt = gof_dt[ind]
        obs = np.log(rsq_dt.Bd_load.values)
        pred = rsq_dt.mean_load.values
        v_pred = np.var(pred)
        v_resid = np.var((obs - pred))
        rsq_alt = (v_pred) / (v_pred + v_resid)

        # R2 results
        rsq_ests[dod_location] = rsq_alt

        # Calculate the number of instances where the observed prevalence
        # is significantly different than predicted prevalence at alpha = 0.05
        # corrected for multiple comparisons.
        gof_dt = gof_dt.assign(month=lambda x: x.date.dt.month,
                               year=lambda x: x.date.dt.year)

        def my_group(x):

            names = {'pred_prev': x.prev.mean(),
                     'n': len(x),
                     'inf': np.sum(x.Bd_pos == 1)}
            return(pd.Series(names, index=['pred_prev', 'n', 'inf']))

        df = gof_dt.groupby(['Site_code', 'date']).apply(my_group).reset_index()

        # Get p-values from binomial test
        pvals = np.array([stats.binom_test(df.inf.values[i],
                                           n=df.n.values[i],
                                           p=df.pred_prev.values[i])
                          for i in range(df.shape[0])])

        # Only compare values with 5 or samples so we have some power to detect
        # differences
        pvals = pvals[df.n.values >= 5]

        # Use Bonferonni correction
        alpha = 0.05  # Type I error rate
        percent_sig = np.sum(pvals < (alpha / len(pvals))) / len(pvals)

        # Prevalence results
        percent_pvals[dod_location] = (percent_sig, len(pvals))


