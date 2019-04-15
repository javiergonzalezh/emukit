import numpy as np

from emukit.core import ParameterSpace
from emukit.core.interfaces import IModel, IDifferentiable
from emukit.core.acquisition import Acquisition
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable
from emukit.bayesian_optimization.acquisitions.local_penalization import LocalPenalization
from emukit.core.optimization import LocalSearchAcquisitionOptimizer
from emukit.bayesian_optimization.local_penalization_calculator import _estimate_lipschitz_constant
from emukit.bayesian_optimization.acquisitions import ExpectedLoss
from quadrature.emin_epmgp import emin_epmgp



from typing import Union


class PredictFutureLocationsLocalPenalization:
    def __init__(self, acquisition: Acquisition, model: Union[IModel, IDifferentiable], parameter_space: ParameterSpace,
                 n_look_ahead: int):
        self.acquisition = acquisition
        self.acquisition_optimizer = LocalSearchAcquisitionOptimizer(parameter_space, num_steps = 200,
                                                                     num_init_points = 15)
        self.n_look_ahead = n_look_ahead
        self.model = model
        self.parameter_space = parameter_space

    def compute_next_points(self, x: np.array, context: dict = None) -> np.ndarray:
        """
        Predicts the behaviour of the policy in future locations using the Local Penalization

        :param x: Initial location in which the policy is evaluated

        """
        self.acquisition.update_parameters()

        # Initialize local penalization acquisition
        local_penalization_acquisition = LocalPenalization(self.model)

        # Everything done in log space so addition here is same as multiplying acquisition with local penalization
        # function.
        acquisition = self.acquisition + local_penalization_acquisition

        future_locations = [x]
        for i in range(self.n_look_ahead - 1):
            # Collect point
            x_next, _ = self.acquisition_optimizer.optimize(acquisition, context)
            future_locations.append(x_next)

            # Update local penalization acquisition with x_next
            f_min = np.min(self.model.Y)
            lipschitz_constant = _estimate_lipschitz_constant(self.parameter_space, self.model)
            local_penalization_acquisition.update_batches(np.concatenate(future_locations, axis=0), lipschitz_constant,
                                                          f_min)
        return np.concatenate(future_locations, axis=0)



class AcquisitionGLASSES(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], parameter_space: ParameterSpace)-> None:
        """
        This acquisition computes the one-step look-ahead myopic expected loss.

        For more information see:

        GLASSES: Relieving The Myopia Of Bayesian Optimisation
        Gonzalez, Javier and Osborne, Michael and Lawrence, Neil.
        19th International Conference on Artificial Intelligence and Statistics (AISTATS)
        2016, Cadiz, Spain. JMLR

        :param model: model that is used to compute the improvement.
        :param jitter: parameter to encourage extra exploration.
        """

        self.model = model
        self.parameter_space = parameter_space
        self.expected_loss = ExpectedLoss(self.model)
        self.remaining_look_ahead = 1

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the non-myopic GLASSES loss.

        :param x: points where the acquisition is evaluated.
        """

        ### --- predict future locations of the policy
        oracle = PredictFutureLocationsLocalPenalization(self.expected_loss, self.model,
                                                        self.parameter_space, self.remaining_look_ahead)
        oracle_locations = oracle.compute_next_points(x)

        ### --- predict the future locations
        mean, _ = self.model.predict(oracle_locations)
        covariance = self.model.predict_covariance(oracle_locations)

        ### --- compute the loss
        y_minimum = np.min(self.model.Y, axis=0)
        glasses_loss = emin_epmgp(mean, covariance, y_minimum)

        return glasses_loss

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return False

