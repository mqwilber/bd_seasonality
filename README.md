# Code, data and models to explore the causes and consequences of seasonality on host-parasite dynamics

This files describes the content of the folder containing the code and data to replicate the analyses presented in the manuscript *Once a reservoir, always a reservoir? Seasonality affects the pathogen maintenance potential of amphibian hosts*. All of the analyses were performed in Python 3.  The Python environment from which all of the results were generated is given by the `environment.yml` file. The Python environment needed to replicate the analyses can be built from this file in the Anaconda platform (see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

**First steps**

1. Unzip `data.zip` at its current location in the directory structure
2. Unzip `results/pickled_results.zip` at its current location in the directory structure
3. Build the Python environment specified by `environment.yml`

`code/`: The folder contains the modules and scripts that simulate the reduced and full host-parasite IPM, fit the model, and explore the dynamics of the resulting model. Documentation is also provided in each script.

- `abc_sampler.py`: A helper module that allows for easy specification of prior probabilities. Used when implementing the MCMC sampler and fitting the reduced IPM to data.

- `ccs_simulation.py`: Script the performs the stochastic simulations of the best fit reduced IPMs to calculate the critical community size (CCS).

- `goodness_of_fit.py`: Performs goodness of fit tests on best fit models and observed prevalence and intensity data.

- `full_model.py`: Module that contains the classes used to instantiate and simulate a full host-parasite IPM.

- `model_analysis.*`: A jupyter notebook and html file that reproduces many of the plots and quantities provided in the manuscript using the fitted reduced IPMs. The HTML file gives an easy, non-interactive view into this file.  The Jupyter Notebook (ipynb) allows for exploration of the models.

- `reduced_mcmc.py`: Script that fits the reduced IPMs to the observed field data.

- `reduced_model.py`: Module that contains the classes used to instantiate and simulate a reduced host-parasite IPM.

- `model_params/`: A folder that contains baseline parameters for each geographic location. Within this folder, there are four other folders: `*_params/` for each of the four locations used in this analysis (LA, TN, PA, VT)

    - `*_params/`: This folder contains two files

        - `*_site_level_params.yml`: File contains information on the sites that were sampled in the given location and the sites that are used when fitting the reduced IPM at that location. In addition, this file contains information on the informative priors for the transmission parameter `trans_beta` that were calculated using a slice likelihood approach as described in Appendix S3 of the manuscript. These informative priors differ for each density assumption.

        - `spp_params_*.yml`: File contains a list of default parameters for the particular leopard frog species in the particular site that are used when fitting the model. **NOTE**: Not all parameters given in this file are used when fitting the model. In addition, the values of parameters that are specifically estimated when fitting the model (e.g., `trans_beta`) are not the same as the values shown in this file.  The values given in this file are placeholders and the estimated values are substituted in when simulating the model.  The life history parameters, however, such as `larval_period`, `breeding_start`, `breeding_end`, `aquatic`, `hibernation`, etc. are all fixed and based on our observations of these species in the field and previously reported values. The pickled files described below contain the fitted parameter values.

`data/`: **NOTE**: Unzip data.zip to access `data/` folder. This folder contains the field data on leopard frogs from four different geographic locations that we use in this analysis. This folder also contains temperature data.

- `leopard_frog_data.csv`: The field data that contains information on sampled leopard frogs through time.

- `leopard_frog_data_metadata.txt`: Text file describes the data columns.
- `site_info.csv`: Additional site info on the locations and areas of particular sites used in the study.

- `temperature_data/`: This file contains two folders.

    - `historical_temperature_data/`: Contains two file types (* indicates LA, TN, PA, or VT)

        - `*_1999-2019_temperatures.csv`: Contains the daily temperature estimates from sites within geographic locations from 1999-2019.  All temperature data was extracted from GridMet (http://www.climatologylab.org/gridmet.html).

        - `*_longterm_avg_temperature.csv`: Contains the historical average temperature for sites within a geographic location over 20 years for each day of the year. All temperature data was extracted from GridMet (http://www.climatologylab.org/gridmet.html).

    - `measured_temperature_data/`: Contains four folders, one for each location

        - `*/`: Each folder contains one file

            - `hobo_water_temperature.csv`: Contains the water temperature for a given site within a geographic location every 30 minutes as recorded by HOBO data loggers in the field.

`results/`: The results folder. This folder contains one folder

- `pickled_results/`: This folder contains a file for each of the 48 models shown in the main text. These files are `*.pkl` files, which means they are pickled Python objects.  These files contain all of the MCMC results for the fitted models and are used to extract the best fit parameters and compare DIC values among fitted models.  The naming nomenclature is described in `model_analysis.ipynb`. **NOTE**: To reduce its size, the pickled results folder is initially zipped.  Unzip this folder to gain access to the fitted models.
