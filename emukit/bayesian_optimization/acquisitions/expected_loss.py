# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Union

from GPyOpt.util.general import get_quantiles
import numpy as np

from ...core.interfaces import IModel, IDifferentiable
from ...core.acquisition import Acquisition


## --- Expected Loss

class ExpectedLoss(Acquisition):

    def __init__(self, model: Union[IModel, IDifferentiable], jitter: float = float(0))-> None:
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
        self.jitter = jitter

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Expected Loss.

        :param x: points where the acquisition is evaluated.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        loss =  y_minimum + (mean - y_minimum) * cdf - standard_deviation * pdf

        return loss

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the Expected Loss and its derivative.

        :param x: locations where the evaluation with gradients is done.
        """

        mean, variance = self.model.predict(x)
        standard_deviation = np.sqrt(variance)

        y_minimum = np.min(self.model.Y, axis=0)

        dmean_dx, dvariance_dx = self.model.get_prediction_gradients(x)
        dstandard_deviation_dx = dvariance_dx / (2 * standard_deviation)

        pdf, cdf, u = get_quantiles(self.jitter, y_minimum, mean, standard_deviation)

        loss =  y_minimum + (mean - y_minimum) * cdf - standard_deviation*pdf
        dloss_dx = dstandard_deviation_dx * pdf - cdf * dmean_dx # Same as EI

        return loss, dloss_dx

    @property
    def has_gradients(self) -> bool:
        """Returns that this acquisition has gradients"""
        return isinstance(self.model, IDifferentiable)