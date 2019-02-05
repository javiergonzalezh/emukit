# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from GPyOpt.acquisitions.base import AcquisitionBase
from GPy.models import GPClassification
import numpy as np

class AcquisitionCEI(AcquisitionBase):
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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionCEI, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.probability_function = lambda x: model.predict(x)[0]

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        :param x: it is a duel
        """
        Cmax = copelandScoreMax(self.probability_function, self.space)

        acq_rlts = np.empty((x.shape[0],1))

        for x_i in range(x.shape[0]):

            # --- duplicate the observation
            x_new = np.empty((2,x.shape[1]))
            x_new[0] = x[x_i:x_i+1]
            prob_y_new = self.model.predict(x[x_i:x_i+1])[0]
            x_new[1,:x.shape[1]/2] = x[x_i,x.shape[1]/2:]
            x_new[1,x.shape[1]/2:] = x[x_i,:x.shape[1]/2]

            # --- compute possible outcomes
            y_positive = np.array([[1],[0]])
            y_negative = np.array([[0],[1]])

            # --- augment X with the new x and Y with the two possible outcomes
            X_augmented = np.vstack((self.model.model.X,x_new))
            Y_augmented_positive = np.vstack((self.model.model.Y,y_positive))
            Y_augmented_negative = np.vstack((self.model.model.Y,y_negative))

            # --- create the the two possible new models with the potential outcomes of the experiment
            model_augmented_positive = self.model.copy()
            model_augmented_negative = self.model.copy()
            model_augmented_positive.updateModel(X_augmented,Y_augmented_positive,X_augmented,Y_augmented_positive)
            model_augmented_negative.updateModel(X_augmented,Y_augmented_negative,X_augmented,Y_augmented_negative)

            pos_Cmax = copelandScoreMax(lambda x: model_augmented_positive.predict(x)[0], self.space)
            neg_Cmax = copelandScoreMax(lambda x: model_augmented_negative.predict(x)[0], self.space)

            # Option 1:
            acq_rlts[x_i] =  max(max(pos_Cmax,neg_Cmax)-Cmax,0)

            # Option 2:
            acq_rlts[x_i] = prob_y_new*max(pos_Cmax-Cmax,0) + (1.-prob_y_new)*max(neg_Cmax-Cmax,0)

        return acq_rlts


class AcquisitionCEI_SVIGP(AcquisitionBase):
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

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionCEI_SVIGP, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.probability_function = lambda x: model.predict(x)[0]

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost
        :param x: it is a duel
        """
        Cmax = copelandScoreMax(self.probability_function, self.space)

        acq_rlts = np.empty((x.shape[0],1))

        for x_i in range(x.shape[0]):

            # --- duplicate the observation
            x_new = np.empty((2,x.shape[1]))
            x_new[0] = x[x_i:x_i+1]
            prob_y_new = self.model.predict(x[x_i:x_i+1])[0]
            x_new[1,:x.shape[1]/2] = x[x_i,x.shape[1]/2:]
            x_new[1,x.shape[1]/2:] = x[x_i,:x.shape[1]/2]

            # --- compute possible outcomes
            y_positive = np.array([[1],[0]])
            y_negative = np.array([[0],[1]])

            # --- augment X with the new x and Y with the two possible outcomes
            X_augmented = np.vstack((self.model.model.X,x_new))
            Y_augmented_positive = np.vstack((self.model.model.Y,y_positive))
            Y_augmented_negative = np.vstack((self.model.model.Y,y_negative))

            # --- create the the two possible new models with the potential outcomes of the experiment
            model_augmented_positive = self.model.copy()
            model_augmented_negative = self.model.copy()
            model_augmented_positive.set_XY(X_augmented,Y_augmented_positive)
            model_augmented_positive.kern.fix(warning=False)
            model_augmented_positive.optimize(max_iters=10)
            model_augmented_negative.set_XY(X_augmented,Y_augmented_negative)
            model_augmented_negative.kern.fix(warning=False)
            model_augmented_negative.optimize(max_iters=10)

            pos_Cmax = copelandScoreMax(lambda x: model_augmented_positive.predict(x)[0], self.space)
            neg_Cmax = copelandScoreMax(lambda x: model_augmented_negative.predict(x)[0], self.space)

            # Option 1:
            acq_rlts[x_i] =  max(max(pos_Cmax,neg_Cmax)-Cmax,0)

            # Option 2:
            acq_rlts[x_i] = prob_y_new*max(pos_Cmax-Cmax,0) + (1.-prob_y_new)*max(neg_Cmax-Cmax,0)

        return acq_rlts


def copelandScore(p_f, space, x):
    """Evaluate the Copeland Score function."""
    arms = space.spaces[0].get_bandit()
    expanded_arms = np.empty((x.shape[0],arms.shape[0],x.shape[1]+arms.shape[1]))
    expanded_arms[:,:,:x.shape[1]] = x[:,None,:]
    expanded_arms[:,:,x.shape[1]:] = arms[None,:,:]
    expanded_arms = expanded_arms.reshape(x.shape[0]*arms.shape[0],x.shape[1]+arms.shape[1])
    # p = (p_f(expanded_arms).reshape(x.shape[0],arms.shape[0])>0.5).mean(1)
    p = p_f(expanded_arms).reshape(x.shape[0],arms.shape[0]).mean(1)
    return p

def copelandScoreMax(p_f, space):
    """Evaluate the Copeland Score function."""
    arms = space.spaces[0].get_bandit()
    c = copelandScore(p_f,space,arms)
    return c.max()