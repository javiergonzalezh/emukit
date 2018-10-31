import numpy as np
import scipy.optimize

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable
from emukit.core.loop import CandidatePointCalculator, LoopState
from emukit.bayesian_optimization.acquisitions.local_penalization import LocalPenalization
from emukit.core.optimization import AcquisitionOptimizer
from emukit.experimental_design.model_free.random_design import RandomDesign

N_SAMPLES = 500  # Number of samples to use when estimating Lipschitz constant
MAX_ITER = 200  # Maximum number of iterations for optimizer when estimating Lipschitz constant


class LocalPenalizationPointCalculator(CandidatePointCalculator):
    """
    Candidate point calculator that computes a batch using local penalization from:

    `Batch Bayesian Optimization via Local Penalization. Javier González, Zhenwen Dai, Philipp Hennig, Neil D. Lawrence
    <https://arxiv.org/abs/1505.08052>`_
    """

    def __init__(self, acquisition: Acquisition, acquisition_optimizer: AcquisitionOptimizer, model: IDifferentiable,
                 parameter_space: ParameterSpace, batch_size: int):
        """
        :param acquisition: Base acquisition function to use without any penalization applied, this should have a log
                            transform applied if the acquisition function can be negative
        :param acquisition_optimizer: AcquisitionOptimizer object to optimize the penalized acquisition
        :param model: Model object, used to compute the parameters of the local penalization
        :param parameter_space: Parameter space describing input domain
        :param batch_size: Number of points to collect in each batch
        """
        self.acquisition = acquisition
        self.acquisition_optimizer = acquisition_optimizer
        self.batch_size = batch_size
        self.model = model
        self.parameter_space = parameter_space

    def compute_next_points(self, loop_state: LoopState) -> np.ndarray:
        """
        Computes a batch of points using local penalization.

        :param loop_state: Object containing the current state of the loop
        """

        # Initialize local penalization acquisition
        local_penalization_acquisition = LocalPenalization(self.model)

        # Everything done in log space so addition here is same as multiplying acquisition with local penalization
        # function.
        acquisition = self.acquisition + local_penalization_acquisition

        x_batch = []
        for i in range(self.batch_size):
            # Collect point
            x_next, _ = self.acquisition_optimizer.optimize(acquisition)
            x_batch.append(x_next)

            # Update local penalization acquisition with x_next
            f_min = np.min(self.model.Y)
            lipschitz_constant = _estimate_lipschitz_constant(self.parameter_space, self.model)
            local_penalization_acquisition.update_batches(np.concatenate(x_batch, axis=0), lipschitz_constant, f_min)
        return np.concatenate(x_batch, axis=0)


def _estimate_lipschitz_constant(space: ParameterSpace, model: IDifferentiable):
    """
    Estimate the lipschitz constant of the function by max norm of gradient currently in the model. Find this max
    gradient norm using an optimizer.
    """
    def gradient_norm(x):
        x = np.atleast_2d(x)
        d_mean_d_x, _ = model.get_prediction_gradients(x)
        result = np.sqrt((np.square(d_mean_d_x)).sum(1))  # simply take the norm of the expectation of the gradient
        return -result

    # Evaluate at some random samples first and start optimizer from point with highest gradient
    random_design = RandomDesign(space)
    samples = random_design.get_samples(N_SAMPLES)
    samples = np.vstack([samples, model.X])
    gradient_norm_at_samples = gradient_norm(samples)
    x0 = samples[np.argmin(gradient_norm_at_samples)]

    # Run optimizer to find point of highest gradient
    res = scipy.optimize.minimize(gradient_norm, x0, bounds=space.convert_to_gpyopt_design_space().get_bounds(),
                                  options={'maxiter': MAX_ITER})
    lipschitz_constant = -res.fun[0]

    if lipschitz_constant < 1e-7:
        # To avoid problems in cases in which the model is flat.
        lipschitz_constant = 10
    return lipschitz_constant
