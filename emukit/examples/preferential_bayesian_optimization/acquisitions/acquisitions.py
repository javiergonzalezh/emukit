# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase
from GPy.models import GPClassification
import numpy as np






class AcqBOPPER_explore(AcquisitionBase):
    """
    Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcqBOPPER_explore, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        :param x: it is a duel
        """
        var_p = self.model.predict_var(x)

        return var_p




class AcqBOPPER_explore(AcquisitionBase):
    analytical_gradient_prediction = False
    def __init__(self, model, space, optimizer=None, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcqBOPPER_explore, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.probability_function = lambda x: model.predict(x)[0]

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        :param x: it is a duel
        """
        var_p = self.model.predict_var(x)
        return var_p[:,None]

class AcqBOPPER_sample_max(AcquisitionBase):
    analytical_gradient_prediction = False
    def __init__(self, model, space, optimizer=None, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcqBOPPER_sample_max, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.probability_function = lambda x: model.predict(x)[0]

    def _compute_acq(self, x):
        # It works because optimizer evaluate ONLY ONCE with ALL THE STATES !!!!!!
        ps = self.model.sample_p(x)
        grid_size = np.sqrt(ps.shape[0])
        ps = ps.reshape(grid_size,grid_size)
        copeland = ps.mean(1)
        cmax_idx = copeland.argmax()
        idx = ps[cmax_idx].argmax()

        v = np.zeros((grid_size,grid_size))
        v[cmax_idx,idx] = 1
        v = v.flatten()

        return v[:,None]

class AcqBOPPER_sample_explore(AcquisitionBase):
    analytical_gradient_prediction = False
    def __init__(self, model, space, optimizer=None, cost_withGradients=None):
        self.optimizer = optimizer
        super(AcqBOPPER_sample_explore, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.probability_function = lambda x: model.predict(x)[0]

    def _compute_acq(self, x):
        # It works because optimizer evaluate ONLY ONCE with ALL THE STATES !!!!!!
        ps = self.model.sample_p(x)
        var_p = self.model.predict_var(x)
        grid_size = np.sqrt(ps.shape[0])
        ps = ps.reshape(grid_size,grid_size)
        var_p = var_p.reshape(grid_size,grid_size)
        copeland = ps.mean(1)
        cmax_idx = copeland.argmax()
        idx = var_p[cmax_idx].argmax()

        v = np.zeros((grid_size,grid_size))
        v[cmax_idx,idx] = 1
        v = v.flatten()

        return v[:,None]