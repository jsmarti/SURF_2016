"""
A simple Gaussian process model.

Author:
    Ilias Bilionis

Date:
    6/1/2015

"""


__all__ = ['GPRegression']


import GPy
import pymc as pm
import numpy as np
import scipy
import math
from . import LogLogistic
from . import Jeffreys
from . import MultivariateNormal
from . import MultivariateNormalDiagonalCovariance
from . import Cholesky
from . import LognormalStepMethod


class GPRegression(object):
    
    """
    Gaussian process regression.

    :param X:   Observed input points (num_obs x num_dim).
    :param Y:   Observed outputs (num_obs x 1).
    :param k:  Kernel for the mean response.
    """

    # Observed input points
    _X = None

    # Observed outputs
    _Y = None

    # Kernel for the mean response
    _k = None

    @property 
    def X(self):
        """
        :getter:    Get the observed input points.
        """
        return self._X

    @property 
    def num_obs(self):
        """
        :getter:    The number of observations.
        """
        return self._X.shape[0]

    @property 
    def input_dim(self):
        """
        :getter:    The number of design dimensions.
        """
        return self._X.shape[1]

    @property 
    def Y(self):
        """
        :getter:    Get the output points.
        """
        return self._Y

    @property 
    def k(self):
        """
        :getter:    The kernel for the mean response.
        """
        return self._k 

    def __init__(self, X, Y, k=None):
        """
        Initialize object. See class docstring for details.
        """
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        self._X = X
        assert isinstance(Y, np.ndarray)
        if Y.ndim == 2:
            assert Y.shape[1] == 1
            Y = Y.flatten()
        assert Y.shape[0] == X.shape[0]
        self._Y = Y
        if k is None:
            k = GPy.kern.RBF(self.input_dim, ARD=True)
        self._k = k

    def make_model(self):
        """
        Returns a dictionary representing the model to be used with PyMC.
        """
        # Hyper-parameters for the mean response theta_m, see Eq. (8) and
        assert isinstance(self.k, GPy.kern.RBF), 'Only working for RBF kernel'
        # Length scales for the mean response, see Eq. (43) and Eq. (46)
        #ell_m = pm.Uniform('ell_m', 0.1, 1., value=np.ones((self.input_dim,)))
        ell = LogLogistic('ell', value=np.ones(self.input_dim,))
        # Signal strength of the mean response, see Eq. (47) for alpha = m
        s = Jeffreys('s', value=1.)
        #s_m = LogLogistic('s_m', value=1.)
        # Jitter of the mean covariance, see Eq. (48) for alpha = m
        j = Jeffreys('j', value=1.)

        # MEAN RESPONSE
        # Correlation matrix
        @pm.deterministic(plot=False, trace=False)
        def C(ell=ell, X=self.X):
            self.k.variance = 1.
            self.k.lengthscale = ell
            return self.k.K(X)
        # Covariance matrix without jitter
        @pm.deterministic(plot=False, trace=False)
        def Knj(s=s, C=C):
            return (s ** 2) * C
        # Covariance matrix with jitter
        @pm.deterministic(plot=False, trace=False)
        def K(j=j, Knj=Knj):
            return Knj + (j ** 2) * np.eye(Knj.shape[0])
        # The Cholesky of the covariance
        U = Cholesky('U', C=K, plot=False, trace=False)
        # The observations
        y = MultivariateNormal('y', value=self.Y, mu=np.zeros((self.num_obs, )),
                               U=U, observed=True)

        return pm.Model(locals())

    def sample(self, num_samples, num_burn=100, num_thin=10):
        """
        Sample from the posterior of the hyper-parameters using MCMC.

        See Sec. 2.3 of the paper.
        """
        model = self.make_model()
        mcmc = pm.MCMC(model)
        mcmc.use_step_method(LognormalStepMethod, model.ell)
        mcmc.use_step_method(LognormalStepMethod, model.s)
        mcmc.use_step_method(LognormalStepMethod, model.j)
        mcmc.sample(num_samples, burn=num_burn, thin=num_thin,
                    tune_throughout=False, verbose=0)
        pm.Matplot.plot(mcmc)
