import numpy as np
import scipy.stats as stats
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt

"""
A generic ABC-SMC sampler.

Takes in a model function, model parameters, an observed data vector, and a
distance function and then performs ABC-SMC to estimate parameters.

Optional arguments
------------------
- transformations
    - Dict specifying the transformations to do on each parameter to place the
    parameter into unconstrained state-space. Options are "none" (default),
    "log" (for strictly positive parameters),
    "logit" (for 0 to 1 parameters), and "minmax" for if you want to specify
    the range.
- priors
    - Dict specifying the prior distributions for each parameter
"""


class MultiABC(object):

    def __init__(self, models, params, observed, dist_fxn,
                 transformations=None, priors=None, model_priors=None, sd=5,
                 num_particles=1000, percent=10, model_kwargs=None):
        """
        Class will fit and compare multiple models using SMC-ABC. Passing a
        single model should yield the same as ABC class.

        Parameters
        ----------
        models : Dict
            Dict model functions. Keys are model names. Each model fxn should be as defined in ABC class.
        params: Dict
            Dict of lists.  Each key is a model name and looks up a list of
            parameter names for each model. As described in ABC class.
        observed : array
            Array of observed data
        dist_fxn : fxn
            Takes in observed vector and predicted vector an returns a
            univariate distance metric. 'observed' and 'predicted' can be
            anything a long as the distance function evaluates them in such a
            way that the function returns a single value.
        transformations : dict
            Keys of dicts are model names.  Each dict looks up a dict of
            transformations as defined in ABC class.
        priors : dict
            Keys of dicts are model names. Each dict looks up a dict of priors
            as defined in ABC class.
        model_priors : dict or None
            If None, assigns all models equal prior probabilities. If dict,
            each key is a model name and looks up a probability. All
            probabilities will be normalized to one.
        sd : float
            Default standard deviation on unconstrained normal parameters
        num_particles : int
            Number of sampled particles to start with.  This will be larger
            than the number of accepted particles
        percent : float (0 - 100)
            The percent of num_particles to keep each round.
            num_particles * (percent / 100) = num accepted particles
        model_kwargs : dict or None
            Keys are model names and look up dictionary of keyword parameters
            to each model function. If None, assumed no keyword parameters. If
            not all models are passed, sets the model keyword parameter of the
            models that are not passed to {}.

        """

        self.model_names = list(models.keys())

        if model_kwargs is None:
            model_kwargs = {mod_nm: {} for mod_nm in self.model_names}
        else:
            kwarg_diff = set(self.model_names) - set(model_kwargs.keys())
            model_kwargs.update({mod_nm: {} for mod_nm in kwarg_diff})


        if transformations is None:
            transformations = {mod_nm: None for mod_nm in self.model_names}

        if priors is None:
            priors = {mod_nm: None for mod_nm in self.model_names}

        if model_priors is None:
            self.model_priors = {mod_nm: 1 / len(self.model_names) for mod_nm
                                 in self.model_names}
        else:
            assert set(self.model_names) == set(model_priors.keys()), "Must give all models a prior weight or set 'model_priors' to None"
            self.model_priors = model_priors

        # For each model, initialize its own ABC class.
        self.abc_models = {mod_nm: ABC(models[mod_nm], params[mod_nm],
                                       observed, dist_fxn,
                                       transformations[mod_nm],
                                       priors[mod_nm], sd=sd,
                                       model_kwargs=model_kwargs[mod_nm])
                           for mod_nm in self.model_names}

        self.num_particles = num_particles
        self.percent = percent
        self.dist_fxn = dist_fxn
        self.observed = observed
        self.model_kwargs = model_kwargs

    def fit(self, steps, sigma=None, njobs=1, print_progress=True,
            logger=None):
        """
        Use ABC-SMC to sample and compare models. Using algorithm from
        Toni et al. 2009, with a tolerance based on the percentile of the
        distance (Kosmala et al. 2015).

        Parameters
        ----------
        steps : int
            The number of steps in the ABC fitting. steps = 1 is a rejection
            algorithm
        sigma : None or float
            Perturbation standard deviation for ABC-SMC. If None uses an
            increasingly shrinking perturbation based on sd of accepted
            particles (Kosmala et al. 2015). Otherwise, uses a fixed sigma
            for all parameters.
        njobs : int
            Number of cores to use for running the model iterations
        print_progress : bool
            Print the progress of the ABC fitting
        logger : Default is None or a logging object
            If not None, logs progress with logging object


        Returns
        -------
        : self
            All results are stored in self.abc_models.

        Notes
        -----

        self.model_percent gives the probability of the ABC approximation
        of P(model | data) for each model. If a model is excluded, the
        approximate probability is < (1 / total models) accepted.

        self.model_indexes is a dict that gives the indexes of the best
        parameters for each round of fitting

        However, see Robert et al. 2011, PNAS for the problems with using
        ABC or model selection. If possible, you should use simulated data to
        validate that the given summary statistics can reliably distinguish
        between your models.

        """

        # If it has not been run yet.
        if not hasattr(self, "all_model_percents"):
            # Draw models from priors
            priors = [self.model_priors[nm] for nm in self.model_names]
            self._instantiate_models(self.model_names, priors)

            all_model_percents = []
            all_model_indexes = []
            start = 0
            stop = start + steps
        else:
            all_model_percents = self.all_model_percents
            all_model_indexes = self.all_model_indexes
            start = len(self.all_model_percents)
            stop = start + steps

        # Multiple model ABC-SMC
        for s in range(start, stop):

            # Simulate models
            for i, mnm in enumerate(self.current_models):

                if print_progress:

                    message = ("Simulating model {0} ({1} of {2})"
                               .format(mnm, i + 1, len(self.current_models)))

                    if logger is None:
                        print(message)
                    else:
                        logger.info(message)

                self.abc_models[mnm].fit(1, sigma=sigma, njobs=njobs,
                                             print_progress=print_progress,
                                             logger=logger,
                                             simulate=True, reweight=False,
                                             external_inds=None)

            # Combine the distances for each model
            all_distances = pd.concat([
                                pd.DataFrame({"model_name": mnm,
                                              "distance": self.abc_models[mnm].stored_distances[s]})
                                for mnm in self.current_models])
            all_distances = all_distances.sort_values(by="distance")

            # Extract the top percent of scores across all models
            model_indexes = extract_indexes(all_distances, self.percent)

            # Summarize the current step and save
            tot_models = np.sum([len(x) for x in model_indexes.values()])
            model_percents = {nm: len(x) / tot_models for nm, x in
                              model_indexes.items()}
            all_model_indexes.append(model_indexes)
            all_model_percents.append(model_percents)

            # Draw next set of models
            current_models = list(model_indexes.keys())
            priors = [self.model_priors[nm] for nm in current_models]
            self._instantiate_models(current_models, priors)

            # Draw and re-weight particles
            for i, mnm in enumerate(self.current_models):

                self.abc_models[mnm].fit(1, sigma=sigma, njobs=njobs,
                                         print_progress=False,
                                         logger=logger, reweight=True,
                                         simulate=False,
                                         external_inds=model_indexes[mnm])

        self.all_model_percents = all_model_percents
        self.all_model_indexes = all_model_indexes

        return(self)

    def _instantiate_models(self, model_names, priors):
        """
        Draw model names from priors, and set up each model with the correct
        number of particles

        Parameters
        ----------
        model_names: array-like
            Model names which to draw
        priors : array-like
            Prior weight on each model name. Will be normalized to one.
        """

        priors = np.array(priors) / np.sum(priors)
        init_models = np.random.choice(model_names, p=priors,
                                       size=self.num_particles)
        current_models, current_counts = np.unique(init_models,
                                                   return_counts=True)

        self.current_models = current_models

        # For each model I need to set the num_particles. This will be
        # different for each model based on how many were drawn.
        for mnm, num in zip(current_models, current_counts):
            self.abc_models[mnm].num_particles = num
            self.abc_models[mnm].percent = 100  # Save all particles


class ABC(object):

    def __init__(self, model, params, observed, dist_fxn,
                 transformations=None, priors=None, sd=5,
                 num_particles=1000, percent=10, model_kwargs={}):
        """
        Parameters
        ----------
        model : function
            function that takes in a dict of parameters and returns a prediction
            in the same format as observed.
        params : list
            names of parameters used in model
        observed : array
            Array of observed data
        dist_fxn : fxn
            Takes in observed vector and predicted vector an returns a
            univariate distance metric. 'observed' and 'predicted' can be
            anything a long as the distance function evaluates them in such a
            way that the function returns a single value.
        transformations : dict
            Indicates which parameters need to be transformed to put them on
            an constrained scale.

            Default: None -> All parameters will assume to be on an unconstrained
            scale

            Dict: keys are parameter names and values are either
                None : No transformation needed
                'log': Log-transformation to put on unconstrained scale
                       (positive values)
                'logit': Logit-transformation to put 0-1 data on unconstrained scale.
                'minmax':
                    Transform variable bounded between lower and upper. The
                    prior distribution will specify the bounds. prior.support()
        priors : dict
            All priors are given on the untransformed scale.

            None: All priors will be assumed to normal, centered 0, with
            a standard deviation of 10.

            Dict: Keys are parameter names and values are frozen scipy
                  distributions.
        sd : float
            Default standard deviation on unconstrained normal parameters
        num_particles : int
            Number of sampled particles to start with.  This will be larger
            than the number of accepted particles
        percent : float (0 - 100)
            The percent of num_particles to keep each round.
            num_particles * (percent / 100) = num accepted particles
        model_kwargs : dict
            The model function can also take keyword arguments. These are fixed
            and will be passed to "model" each time it is called.
        """

        self.model = model
        self.observed = observed
        self.dist_fxn = dist_fxn
        self.params = params
        self.model_kwargs = model_kwargs

        # ABC params
        self.num_particles = num_particles
        self.percent = percent
        self.maxkeep = np.int(np.floor(self.num_particles * (self.percent / 100.)))

        # By default assume zero-centered, vague priors
        self.priors = {p: stats.norm(loc=0, scale=sd) for p in params}
        if priors is not None:
            self.priors.update(priors)

        # Contains transformation and inverse
        self.transformations = {p: (_identity, _identity) for p in params}
        self.jacobians = {p: _identity_jac for p in params}
        if transformations is not None:
            for p, trans in transformations.items():
                if trans == "logit":
                    self.transformations[p] = (_logit, _inv_logit)
                    self.jacobians[p] = _logit_jac
                elif trans == "log":
                    self.transformations[p] = (_log, _exp)
                    self.jacobians[p] = _exp
                elif trans == "minmax":
                    self.transformations[p] = (_minmax, _inv_minmax)
                    self.jacobians[p] = _minmax_jac

        # Get the support for each parameter.  Only used for minmax.
        self.bounds = {p: self.priors[p].support() for p in self.priors}

    def fit(self, steps, sigma=None, njobs=1, print_progress=True,
            logger=None, simulate=True, reweight=True, external_inds=None):
        """

        Parameters
        ----------
        steps : int
            The number of steps in the ABC fitting. steps = 1 is a rejection
            algorithm
        sigma : None or float
            Perturbation standard deviation for ABC-SMC. If None uses an
            increasingly shrinking perturbation based on sd of accepted
            particles (Kosmala et al. 2015). Otherwise, uses a fixed sigma
            for all parameters.
        njobs : int
            Number of cores to use for running the model iterations
        print_progress : bool
            Print the progress of the ABC fitting
        logger : Default is None or a logging object
            If not None, logs progress with logging object
        simulate : bool
            Used when processing multiple models. Keep as True. If
            False, the assumption is that self.stored_distance already contains
            distances and the model simulation doesn't need to be run.
        reweight : bool
            Used when processing multiple models. Keep as True. If False,
            no reweighting will be done after model simulation.
        external_inds : array-like or None
            Pass in an external array of indexes that should be kept when
            selecting the best model.  Used for multi-model comparison.

        Returns
        -------
        : dict
            'params': Accepted parameters for each step
            'weights': Parameter weights for each step
            'sigmas': Parameter sigmas for each step

        Notes
        -----
        TODO: Allow for flexibility in perturbation distribution
        """

        assert (sigma is None) | ((type(sigma) == float) | (type(sigma) == int)), "sigmas must be None or float"
        if (self.percent < 1):
            print("Message: Checking...did you mean for percent to be < 1?")

        # Check if self.stored_particles already exists
        # If not, initialize
        if not hasattr(self, "stored_particles"):

            stored_weights = []
            stored_sigmas = []
            stored_particles = []
            accepted_particles = []
            stored_distances = []
            accepted_distances = []
            full_distances = []

            # Fixed perturbation
            if sigma is not None:
                fixed_sigmas = np.repeat(sigma, len(self.params))

            # Draw parameters
            init_params = np.array([self.priors[p].rvs(self.num_particles)
                                    for p in self.params]).T

            stored_particles.append(init_params)
            start = 0
            stop = steps
        else:
            # Restart ABC from the previous place it stopped
            stored_weights = self.stored_weights
            stored_sigmas = self.stored_sigmas
            stored_particles = self.stored_particles
            accepted_particles = self.accepted_particles
            stored_distances = self.stored_distances
            accepted_distances = self.accepted_distances
            full_distances = self.full_distances

            if external_inds is None:
                start = len(stored_particles) - 1
            else:
                start = len(stored_weights)
            stop = start + steps

        # Begin ABC algorithm
        for t in range(start, stop):

            if print_progress:

                if not logger:
                    print("Working on step {0}".format(t))
                else:
                    logger.info("Working on step {0}".format(t))

            if simulate:

                # Fit models and compute distances
                params_dict = [{p: v for p, v in zip(self.params, values)}
                               for values in stored_particles[t]]
                fdistances = self.multiprocess_model(njobs, params_dict,
                                                    print_progress, logger)
                full_distances.append(fdistances)

                # If d in distances is more than length 1, sum
                # For checking how distances are converging
                distances = np.array([np.sum(d) for d in fdistances])

                stored_distances.append(distances)

            ## Re-weighting step
            if reweight:

                # Check if some external criteria is provide for keeping
                # particles. Used with multi-model comparison.
                if external_inds is None:
                    # Select subset of particles and store them
                    maxscore = stats.scoreatpercentile(distances, self.percent)
                    # Ensure you only keep the right amount when there are duplicates
                    keep_inds = np.where(distances <= maxscore)[0][:self.maxkeep]
                else:
                    keep_inds = external_inds

                current_particles = stored_particles[t][keep_inds, :]

                accepted_particles.append(current_particles)
                accepted_distances.append(distances[keep_inds])

                # Get the perturbation standard deviation
                trans_particles = np.array([self.transformations[p][0](current_particles[:, i],
                                            lower=self.bounds[p][0], upper=self.bounds[p][1])
                                            for i, p in enumerate(self.params)]).T

                # TODO: Perturbations could also be based on MVN covariance
                # At the cost of taking longer to compute the re-weighting
                # this could reduce the number of necessary simulations.
                if sigma is None:
                    # Beaumont et al. 2009 suggests 2 times the variance

                    # For model comparison there may be only one particle
                    tsigmas = trans_particles.std(axis=0)
                    tsigmas[tsigmas == 0] = 0.01
                    stored_sigmas.append(tsigmas)
                else:
                    stored_sigmas.append(fixed_sigmas)

                # Weight particles with the prior and correction
                prior_weights = self.prior_weights(current_particles)

                if t == 0:
                    renormalize_weights = np.repeat(len(prior_weights), 1)
                else:
                    renormalize_weights = self.renormalize_weights(
                                                current_particles,
                                                accepted_particles[t - 1],
                                                stored_weights[t - 1],
                                                stored_sigmas[t - 1])

                particle_weights = prior_weights / renormalize_weights
                particle_weights = particle_weights / np.sum(particle_weights)
                stored_weights.append(particle_weights)

                # Perturb particles
                pert_particles = self.perturb_particles(current_particles,
                                                        particle_weights,
                                                        stored_sigmas[t])
                stored_particles.append(pert_particles)

        # Store the ABC data
        self.stored_particles = stored_particles
        self.accepted_particles = accepted_particles
        self.stored_sigmas = stored_sigmas
        self.stored_distances = stored_distances
        self.accepted_distances = accepted_distances
        self.stored_weights = stored_weights
        self.full_distances = full_distances

        return({'params': accepted_particles,
                'weights': stored_weights,
                'sigmas':  stored_sigmas,
                'distances': accepted_distances})

    def prior_weights(self, particle_matrix):
        """
        Weight each row of the particle matrix from the prior.
        Assuming independence of each parameter in a multivariate particle.

        Parameters
        ----------
        particle_matrix : N x p matrix
            N is the number of particles and p is the number of parameters

        Returns
        -------
        : array
            The weights of each particle
        """

        logweights = np.array([self.priors[p].logpdf(pvect) + np.log(np.abs(self.jacobians[p](
                                                                            self.transformations[p][0](pvect,
                                                                            lower=self.bounds[p][0], upper=self.bounds[p][1]),
                                                                            lower=self.bounds[p][0], upper=self.bounds[p][1])))
                               for p, pvect in
                               zip(self.params, particle_matrix.T)]).T
        logweights = logweights.sum(axis=1)
        return(np.exp(logweights))

    def renormalize_weights(self, current_particles, past_particles,
                            past_weights, past_sigmas):
        """
        This is the importance re-weighting of the particle to account for the
        fact that we are not drawing from the prior distribution.

        Parameters
        ----------
        current_particles : array-like
            The current matrix of accepted particles
        past_particles : array-like
            The matrix of accepted particles from the previous time step
        past_weights : array-like
            The matrix of weights from the previous particles
        past_sigmas : array-like
            The past sigmas used when perturbing a particle, same length as
            parameters
        """
        # loop through each parameter and build a matrix
        allKs = np.empty((len(self.params), len(past_particles), len(current_particles)))
        for i, p in enumerate(self.params):

            newXX, oldYY = np.meshgrid(current_particles[:, i],
                                       past_particles[:, i])
            tnewXX = self.transformations[p][0](newXX, lower=self.bounds[p][0], upper=self.bounds[p][1])
            toldYY = self.transformations[p][0](oldYY, lower=self.bounds[p][0], upper=self.bounds[p][1])

            K = self.perturbation_pdf(tnewXX, toldYY, past_sigmas[i])
            allKs[i, :, :] = K

        weight_mat = np.tile(past_weights[:, np.newaxis], len(current_particles))
        up_weights = (weight_mat * allKs.prod(axis=0)).sum(axis=0)
        return(up_weights)

    def perturbation_pdf(self, new_x, old_x, sigma):
        """  PDF of the perturbation kernel """
        return(stats.norm(loc=old_x, scale=sigma).pdf(new_x))

    def perturb_particles(self, current_particles, weights, sigmas):
        """
        Draw new particles by drawing new particles based on updated weights
        and then perturbing these parameters based on a Normal kernel with
        sigma sd.

        Parameters
        ----------
        current_particles : array
            The accepted most recent particles.
        weights : array
            The weights of the current_particles
        sigmas : array
            The standard deviations on the transformed scale of each parameter
            in the current_particles

        Returns
        -------
        : peturbed_parameters
            Array of num_particles X p parameters
        """

        ids = np.arange(len(weights))
        new_particle_ids = np.random.choice(ids, size=self.num_particles, p=weights)

        temp_particles = current_particles[new_particle_ids, :]
        perturbed_particles = np.empty(temp_particles.shape)

        # Perturb each parameter. Perturb on the transformed scale and then
        # back transform.
        for i, p in enumerate(self.params):

            mu = self.transformations[p][0](temp_particles[:, i], lower=self.bounds[p][0], upper=self.bounds[p][1])
            tp = stats.norm(loc=mu,
                            scale=sigmas[i]).rvs(size=self.num_particles)

            # Inverse transformation
            perturbed_particles[:, i] = self.transformations[p][1](tp, lower=self.bounds[p][0], upper=self.bounds[p][1])

        return(perturbed_particles)

    def model_mp(self, i, params, print_progress=True, logger=None):
        """
        Evaluate model and compute distance.

        Also returns an index for asynchronous parallelization.
        i is just an integer.

        Parameters
        ----------
        i : int
            Integer for tracking runs in parallelization
        params : dict
            Model parameters

        Returns
        -------
        : Float
            Distance between model predictions and observed
        """

        if print_progress:
            if i % 100 == 0:
                if not logger:
                    print("Simulation {0}".format(i))
                else:
                    logger.info("Simulation {0}".format(i))

        model_res = self.model(params, **self.model_kwargs)
        dist = self.dist_fxn(self.observed, model_res)
        return((i, dist))

    def multiprocess_model(self, processes, params, print_progress, logger=None):
        """
        Multiprocess the simulations of the model

        Parameters
        ----------
        processes : int
            The number of cores to use for multiprocessing
        params : list of dicts
            Each dict contains the parameters for the model

        Returns
        -------
        : list
            List of model output for each parameter set
        """

        if processes <= 1:

            distances = [self.dist_fxn(self.observed,
                                       self.model(p, **self.model_kwargs)) for
                         p in params]

        else:
            pool = mp.Pool(processes=processes)

            results = [pool.apply_async(self.model_mp, args=(i, p, print_progress, logger)) for i, p in
                       enumerate(params)]
            results = [p.get() for p in results]
            results.sort()
            pool.close()

            _, distances = zip(*results)

        return(distances)

    def plot_iterations(self, params=None, figsize=(6, 4),
                              iternums=None, text_size=8):
        """
        Plot parameter estimates and posterior distributions over the ABC
        iterations.

        Parameters
        ----------
        params : list
            List of parameter names to plot. If None plots all parameters.
        figsize : tuple
            Figure size
        iternums : None or array-like
            The iterations numbers to plot. For example, np.array([2, 3, 4])
            will plot the parameter distributions for iteration 2, 3 and 4

        Returns
        -------
        : axes
            Flattened array of plot axes

        """

        if params is None:
            params = self.params

        for p in params:
            assert p in self.params, "{0} is not a parameter".format(p)

        # Check if model has been fit
        if hasattr(self, "accepted_particles"):

            if len(params) <= 3:
                rows = 1
                cols = len(params)
            else:
                modp = np.int(np.ceil(len(params) / 3))
                rows = modp
                cols = 3

            fig, axes = plt.subplots(rows, cols, sharex=False, figsize=figsize)

            try:
                axes = axes.ravel()
            except AttributeError:
                axes = [axes]

            fit = self.accepted_particles
            for p in range(len(params)):

                prms = []
                for i in range(len(fit)):
                    ind = np.where(np.array(self.params) == params[p])[0][0]
                    prms.append(fit[i][:, ind])

                prms = np.array(prms)
                alliters = np.arange(len(fit))
                if iternums is None:
                    iternums = alliters

                axes[p].boxplot([p for p in prms[iternums, :]], positions=alliters[iternums])
                axes[p].set_title(params[p], size=text_size)
                axes[p].tick_params(labelsize=text_size - 2)

                if axes[p].is_first_col():
                    axes[p].set_ylabel("Parameter value", size=text_size)
                if axes[p].is_last_row():
                    axes[p].set_xlabel("ABC iteration", size=text_size)

            for i, ax in enumerate(axes):

                if i >= len(params):
                    axes[i].tick_params(labelsize=0, bottom=False, left=False)
                    axes[i].spines['right'].set_visible(False)
                    axes[i].spines['top'].set_visible(False)
                    axes[i].spines['bottom'].set_visible(False)
                    axes[i].spines['left'].set_visible(False)

            plt.tight_layout()
            return(axes)

        else:
            raise(AttributeError("Model has not yet been fit"))

    def plot_distances(self):
        """
        Diagnostic plot of distances for models at each iteration.

        Returns
        -------
        : ax

        Notes
        -----
        Ideally, you want to see the distances decreasing over iterations and
        eventually leveling off.

        Error bars are the inter quartile range.

        """

        if hasattr(self, "stored_distances"):

            distances_med = []
            distances_lower = []
            distances_upper = []
            all_distances = []

            for d in self.accepted_distances:
                distances_med.append(np.median(d))
                distances_lower.append(stats.scoreatpercentile(d, 25))
                distances_upper.append(stats.scoreatpercentile(d, 75))

            med = np.array(distances_med)
            lower = np.array(distances_lower)
            upper = np.array(distances_upper)
            vals = np.arange(len(self.stored_distances))
            fig, ax = plt.subplots(1, 1)
            #ax.plot(vals, med, '-')
            ax.errorbar(vals, med, yerr=[med - lower, upper - med], marker='o')
            ax.set_xlabel("ABC iterations")
            ax.set_ylabel("Mean distance")

            return(ax)

        else:
            raise(AttributeError("Model has not yet been fit"))


def _exp(x, lower=None, upper=None):
    return(np.exp(x))


def _log(x, lower=None, upper=None):
    return(np.log(x))


def _identity(x, lower=None, upper=None):
    return(x)


def _identity_jac(x, lower=None, upper=None):
    return(1)


def _logit(x, lower=None, upper=None):
    return(np.log(x / (1 - x)))


def _inv_logit(x, lower=None, upper=None, tol=1e-10):

    prob = 1 / (1 + np.exp(-x))
    prob[prob == 1] = 1 - tol
    prob[prob == 0] = tol
    return(prob)


def _logit_jac(x, lower=None, upper=None):
    return(np.exp(-x) / (1 + np.exp(-x))**2)


def _minmax(x, lower=0, upper=1):
    return(np.log((x - lower) / (upper - x)))


def _inv_minmax(x, lower=0, upper=1, tol=1e-10):

    trans = (np.exp(x)*upper + lower) / (1 + np.exp(x))
    trans[trans == lower] = lower + tol
    trans[trans == upper] = upper - tol
    return(trans)


def _minmax_jac(x, lower=0, upper=1):
    return((np.exp(-x)*(upper - lower)) / (1 + np.exp(-x))**2)


def extract_indexes(distances, percent):
    """
    Combine all distances and extract the smallest percent

    Parameters
    ----------
    distances: DataFrame
        columns: model_name - Unique model names
                 distance - Distance for a given parameter set
    percent: float
        The percent of the smallest distances to keep

    Returns
    -------
    : dict
        Keys are model names, values are arrays with indexes to keep for each
        model

    """
    maxscore = stats.scoreatpercentile(distances.distance, percent)
    keep_dist = distances[distances.distance <= maxscore]
    model_indexes = {mnm: (keep_dist[keep_dist.model_name == mnm]
                           .index.values)
                     for mnm in keep_dist.model_name.unique()}
    return(model_indexes)


