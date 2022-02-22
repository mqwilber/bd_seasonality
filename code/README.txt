`code/`: The folder contains the modules and scripts that simulate the reduced and full host-parasite IPM, fit the model, and explore the dynamics of the resulting model. Documentation is also provided in each script. Please see clone https://github.com/mqwilber/bd_seasonality for the full repository.

    - `abc_sampler.py`: A helper module that allows for easy specification of prior probabilities. Used when implementing the MCMC sampler and fitting the reduced IPM to data.

    - `ccs_simulation.py`: Script the performs the stochastic simulations of the best fit reduced IPMs to calculate the critical community size (CCS).

    - `goodness_of_fit.py`: Performs goodness of fit tests on best fit models and observed prevalence and intensity data.

    - `full_model.py`: Module that contains the classes used to instantiate and simulate a full host-parasite IPM.

    - `model_analysis.*`: A jupyter notebook and html file that reproduces many of the plots and quantities provided in the manuscript using the fitted reduced IPMs.

    - `reduced_mcmc.py`: Script that fits the reduced IPMs to the observed field data.

    - `reduced_model.py`: Module that contains the classes used to instantiate and simulate a reduced host-parasite IPM.

    - `model_params/`: A folder that contains baseline parameters for each geographic location. Within this folder, there are four other folders: `*_params/` for each of the four locations used in this analysis (LA, TN, PA, VT)

        - `*_params/`: This folder contains two files

            - `*_site_level_params.yml`: File contains information on the sites that were sampled in the given location and the sites that are used when fitting the reduced IPM at that location. In addition, this file contains information on the informative priors for the transmission parameter trans_beta that were calculated using a slice likelihood approach as described in the supplementary material (Appendix S3). These informative priors differ for each density assumption.

            - `spp_params_*.yml`: File contains a list of default parameters for the particular leopard frog species in the particular site that are used when fitting the model. **NOTE**: Not all parameters given in this file are used when fitting the model. In addition, the values of parameters that are specifically estimated when fitting the model (e.g. trans_beta) are not the same as the values shown in this file.  The values given in this file are place-holders and the estimated values are substituted in when simulating the model.  The life-history parameters, however, such as larval_period, breeding_start, breeding_end, aquatic, hibernation, etc. are all fixed and based on our observations of these species in the field and previously reported values.
