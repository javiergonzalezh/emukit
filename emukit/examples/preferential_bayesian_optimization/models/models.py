# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.models import BOModel
import numpy as np
import GPy

class GPModel_clf(BOModel):
    """
    General class for handling a Gaussain Process classification in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum nunber of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradiens of acquisitions are computable.

    def __init__(self, kernel=None, optimizer='bfgs', max_iters=10, optimize_restarts=5, lengthscale_search_rate=0.8, verbose=True, sample_p_approx=False):
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.lengthscale_search_rate = lengthscale_search_rate
        self.verbose = verbose
        self.sample_p_approx = sample_p_approx
        self.model = None

    def copy(self):
        return GPModel_clf_sklearn(kernel=self.kernel, optimizer=self.optimizer, max_iters=self.max_iters, optimize_restarts=self.optimize_restarts, verbose=self.verbose)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            # kern = GPy.kern.Exponential(self.input_dim, variance=1.) #+ GPy.kern.Bias(self.input_dim)
            # kern.lengthscale.set_prior(GPy.priors.gamma_from_EV(1.,1.))
            kern = GPy.kern.RBF(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        self.model = GPy.models.GPClassification(X, Y, kernel=kern)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        X_all_dual = np.empty((X_all.shape[0]*2,X_all.shape[1]))
        Y_all_dual = np.empty((Y_all.shape[0]*2,Y_all.shape[1]))
        X_all_dual[:X_all.shape[0]] = X_all
        X_all_dual[X_all.shape[0]:,X_all.shape[1]/2:] = X_all[:,:X_all.shape[1]/2]
        X_all_dual[X_all.shape[0]:,:X_all.shape[1]/2] = X_all[:,X_all.shape[1]/2:]
        Y_all_dual[:Y_all.shape[0]] = Y_all
        Y_all_dual[Y_all.shape[0]:] = 1-Y_all
        X_all = X_all_dual
        Y_all = Y_all_dual

        self._create_model(X_all, Y_all)   # --- we do this because the set_XY doesn't work

        l_init = X_all.std(0).max()
        l = l_init
        self.model.kern.lengthscale[:] = l
        self.model.optimize(max_iters=self.max_iters)
        while self.model.kern.variance<1e-3 and l>1e-2*l_init:
            l *= self.lengthscale_search_rate
            self._create_model(X_all, Y_all)
            self.model.kern.lengthscale[:] = l
            self.model.kern.variance[:] = 1.
            for i in range(self.max_iters):
                self.model.optimize(max_iters=10)
        if l<=1e-2*l_init:
            self._create_model(X_all, Y_all)
            self.model.kern.lengthscale[:] = X_all.std(0).max()/4.
            self.model.kern.lengthscale.fix()
            self.model.kern.variance[:] = 1.
            for i in range(self.max_iters):
                self.model.optimize(max_iters=10)

        # # --- update the model maximizing the marginal likelihood.
        # if self.optimize_restarts==1:
        #     self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)
        # else:
        #     self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def predict(self, X):
        """
        Pr editions of the probabilities of the model at X.
        """
        if X.ndim==1: X = X[:,None]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(m*(1-m))

    def sample_p(self,X):
        if X.ndim==1: X = X[None,:]

        if self.sample_p_approx:
            from GPy.inference.latent_function_inference.posterior_sampling import GP_cont_approx
            appx = GP_cont_approx(self.model,1000)
            f = appx.sample_posterior_f()
            fs = f(X)
        else:
            from GPy.util.linalg import jitchol
            fm,fv = self.model._raw_predict(X,full_cov=True)
            L = jitchol
            fs = fm[:,0]+ L.dot(np.random.randn(fm.shape[0]))
            # fs = np.random.multivariate_normal(mean=fm[:,0],cov=fv)
        pm = self.model.likelihood.gp_link.transf(fs)
        return pm

    def predict_var(self,X):
        if X.ndim==1: X = X[None,:]
        degree = 7
        locs, weights = np.polynomial.hermite.hermgauss(degree)
        locs *= np.sqrt(2.)
        weights*= 1./np.sqrt(np.pi)

        fm,fv = self.model._raw_predict(X)
        E_p = self.model.predict(X)[0]
        s = fm+locs[None,:]*np.sqrt(fv)
        E_p2 = ((self.model.likelihood.gp_link.transf(s)**2)*weights[None,:]).sum(1)
        var_p = E_p2 - np.square(E_p[:,0])
        return var_p

    def predict_latent(self,X):
        """
        Predictions of the values of the latent function at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        pass

    def predict_latent_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m, np.sqrt(v), dmdx, dsdx

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names()

class GPModel_clf_sklearn(BOModel):
    """
    General class for handling a Gaussain Process classification in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum nunber of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradiens of acquisitions are computable.

    def __init__(self, kernel=None, optimizer='bfgs', max_iters=1000, optimize_restarts=5,  verbose=True):
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None

    def copy(self):
        m = GPModel_clf_sklearn(kernel=self.model.kernel_.clone_with_theta(self.model.kernel_.theta), optimizer=self.optimizer, max_iters=self.max_iters, optimize_restarts=self.optimize_restarts, verbose=self.verbose)
        return m

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        from sklearn.gaussian_process import GaussianProcessClassifier,kernels

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = kernels.Matern(nu=2.5)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        self.model = GaussianProcessClassifier(kernel=kern,n_restarts_optimizer=self.optimize_restarts)
        self.model.fit(X,Y)
        self.model.X = X
        self.model.Y = Y

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)   # --- we do this because the set_XY doesn't work
        else:
            self.model.fit(X_all, Y_all)

    def predict(self, X):
        """
        Pr editions of the probabilities of the model at X.
        """
        if X.ndim==1: X = X[None,:]
        m = self.model.predict_proba(X)
        m = m[:,1]
        return m, np.sqrt(m*(1-m))

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        pass

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        params = self.model.get_params()
        return np.atleast_2d(np.array(params.values))

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        params = self.model.get_params()
        return params.keys()

class SVM_sklearn(BOModel):
    """
    General class for handling a Gaussain Process classification in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum nunber of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradiens of acquisitions are computable.

    def __init__(self, kernel=None, optimizer='bfgs', max_iters=1000, optimize_restarts=5,  verbose=False):
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None

    def copy(self):
        return SVM_sklearn(kernel=self.kernel, optimizer=self.optimizer, max_iters=self.max_iters, optimize_restarts=self.optimize_restarts, verbose=self.verbose)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        from sklearn.svm import SVC

        # --- define model
        self.model = SVC(C=1e-2, gamma = 10., probability=True, verbose=self.verbose)
        self.model.fit(X,Y)
        self.model.X = X
        self.model.Y = Y

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        self._create_model(X_all, Y_all)   # --- we do this because the set_XY doesn't work

    def predict(self, X):
        """
        Pr editions of the probabilities of the model at X.
        """
        if X.ndim==1: X = X[None,:]
        m = self.model.predict_proba(X)
        m = m[:,1]
        return m, np.sqrt(m*(1-m))

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        pass

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        params = self.model.get_params()
        return np.atleast_2d(np.array(params.values))

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        params = self.model.get_params()
        return params.keys()

class GPModel_clf_svi(BOModel):
    """
    General class for handling a Gaussain Process classification in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum nunber of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = False  # --- Needed in all models to check is the gradiens of acquisitions are computable.

    def __init__(self, inducing_num, inducing_inputs=None, kernel=None, optimizer='bfgs', max_iters=1000, optimize_restarts=1,  verbose=True):
        self.inducing_num = inducing_num
        self.inducing_inputs = inducing_inputs
        self.kernel = kernel
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None

    def copy(self):
        m = GPy.core.SVGP(np.array(self.model.X).copy(),np.array(self.model.Y).copy(),np.array(self.model.Z).copy(),kernel=self.model.kern.copy(),likelihood=GPy.likelihoods.Bernoulli())
        m[:] = self.model[:]
        m.Z.fix(warning=False)
        return m

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Exponential(self.input_dim, variance=1.)
            kern.lengthscale.set_prior(GPy.priors.gamma_from_EV(1.,1.))
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        self.model = GPy.core.SVGP(X,Y,self.inducing_inputs.copy(),kernel=kern,likelihood=GPy.likelihoods.Bernoulli())
        self.model.Z.fix(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        self.model.kern.fix(warning=False)
        self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters,messages=1)
        self.model.kern.unfix()
        self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters,messages=1)

    def predict(self, X):
        """
        Pr editions of the probabilities of the model at X.
        """
        if X.ndim==1: X = X[None,:]
        m, _ = self.model.predict(X)
        return m, np.sqrt(m*(1-m))

    def predict_latent(self,X):
        """
        Predictions of the values of the latent function at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        pass

    def predict_latent_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict_noiseless(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m, np.sqrt(v), dmdx, dsdx

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names()